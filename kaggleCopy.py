import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fashion_mnist = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')


# Check the number of unique labels
print(f'Number of unique labels: {fashion_mnist["label"].nunique()}')

# Create a dictionary to map labels to clothing items
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Check the shape of the dataset
print(f'Dataset shape: {fashion_mnist.shape}\n')

# Check for missing values
print(f'Missing values in dataset: {fashion_mnist.isnull().sum().sum()}\n')


plt.gray()  # B/W Images
plt.figure(figsize=(10, 9))  # Adjusting figure size
plt.subplots_adjust(hspace=0.5) # Space

# Displaying a grid of 3x3 images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = fashion_mnist.iloc[i, 1:].values.reshape(28, 28)
    plt.imshow(img)
    plt.title(label_dict[fashion_mnist.iloc[i, 0]])  # Use the dictionary to display the clothing item name

plt.show()
           



from sklearn.model_selection import train_test_split

# Separate the features and the labels
X = fashion_mnist.iloc[:, 1:]
y = fashion_mnist.iloc[:, 0]

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Training set: {X_train.shape[0]} examples')
print(f'Test set: {X_test.shape[0]} examples')



from sklearn.decomposition import PCA
import numpy as np


# Create a PCA object without specifying the number of components
pca = PCA()

# Fit the PCA model to the data
pca.fit(X_train)

# Get the explained variance ratios
explained_variance_ratios = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratios)

# Find the number of components needed to explain at least 95% of the variance
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1

print(f'Number of components needed to explain 95% of the variance: {n_components_95}')



# Create a PCA object with 187 components
pca = PCA(n_components=n_components_95)

# Fit the PCA model to the training data and apply it to both the training and test data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f'Training set after PCA: {X_train_pca.shape}')
print(f'Test set after PCA: {X_test_pca.shape}')



# Reset the index of y_train if it's a pandas Series or DataFrame
if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.reset_index(drop=True)

# Reverse the PCA transformation for the first 11 images
X_train_reversed = pca.inverse_transform(X_train_pca[:11])

# Visualize the reversed images as a grid of 3x3
plt.figure(figsize=(10, 9))  # Adjusting figure size
plt.subplots_adjust(hspace=0.5) # Space

for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = X_train_reversed[i].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(label_dict[y_train[i]])  # Use the dictionary to display the clothing item name

plt.show()





from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train_pca)

def assign_cluster_labels(kmeans, y_train):
    cluster_labels = {}
    for i in range(len(np.unique(kmeans.labels_))):
        cluster_samples = np.where(kmeans.labels_ == i)[0]
        cluster_labels[i] = y_train[cluster_samples].mode()[0]
    return cluster_labels

cluster_labels = assign_cluster_labels(kmeans, y_train)
print(cluster_labels)




predicted_labels = np.zeros_like(y_train)

for i in range(len(kmeans.labels_)):
    predicted_labels[i] = cluster_labels[kmeans.labels_[i]]

print(predicted_labels[:30].astype(np.uint8))
print('', *y_train[:30], sep=' ')



from sklearn.metrics import accuracy_score
print(f'Cluster analysis accuracy: {accuracy_score(predicted_labels, y_train)}')



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(X_train_pca, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test_pca)

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Get the score of the KNN model
score = knn.score(X_test_pca, y_test)
print(f'Score: {score}')

# Calculate the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate the balanced accuracy
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
print(f'Balanced Accuracy: {balanced_accuracy:.4}')