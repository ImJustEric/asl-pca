"""This is where our main experiment will be run"""
from pca import *
from cnn import *
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from visualize import plot_pca_first_two_components, plot_pca_first_three_components, plot_pca_component_images

# DIRECTORIES, PARAMETERS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "data/filtered")
num_pcs = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000] # Number of PCs to test 

### Load data
images, labels = load_images_and_labels(ROOT_DIR)
y = np.array(labels)
# Split RAW images (so we can preprocess only for PCA, keep raw images for CNN) 
# (80/20), stratify to make sure equal splits across all chars)
images_train, images_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y, random_state=42)
# PCA pipeline (flattened)
X_train = preprocess_images(images_train)
X_test = preprocess_images(images_test)
print(f"Data loaded and preprocessed")


### Run PCA on images 
# Perform cross-validation to see which value of number of principal components is best
# Comment out this block if it takes too long! 
# NOTE: Found that 200 was best, might be higher 
# cv_results = cross_validate_p(X_train, y_train, num_pcs=num_pcs, k=3)
# best_p = max(cv_results, key=cv_results.get)
# print(f"Performed cross validation with result of {best_p}")

# # # If we want to see the accuracies of each num_pcs
# for num, acc in cv_results.items():
#     print(f"Accuracy with {num} principal components: {acc}")

# Now test the best_pc on test data
best_p = 200                            # I have this right now since I tested that 200 was the best out of what I tried
pca = perform_pca(X_train, best_p)
X_train_pca = project_data(pca, X_train)
X_test_pca = project_data(pca, X_test)
print("Projected test data on PCs")

# model = train_knn(X_train_pca, y_train, k=3)
# test_acc = evaluate_model(model, X_test_pca, y_test)

# print(f"Size of testing dataset: {X_test_pca.shape[0]} by {X_test_pca.shape[1]}")
# print(f"Test accuracy under {best_p} principal components: {test_acc}")

# Plot components (uncomment if want plot)
# plot_pca_first_two_components(X_train_pca, y_train, title="Train set: PC1 vs PC2")
# plot_pca_first_three_components(X_train_pca, y_train, title="Train set: PC1 vs PC2 vs PC3")
# plot_pca_component_images(pca, n_components=10, image_shape=(400,400))


# ### Now test against ResNet50 model embeddings, use KNN as well
resnet = load_resnet50()
transform = get_resnet_transform()

X_train_resnet = extract_resnet_features(images_train, resnet, transform)
X_test_resnet = extract_resnet_features(images_test, resnet, transform)

# Plot t-SNE of CNN embeddings (train set)
plot_resnet_tsne(X_train_resnet, y_train)

# Use KNN with CNN embeddings to test accuracy
resnet_knn = train_knn(X_train_resnet, y_train, k=3)
resnet_acc = evaluate_model(resnet_knn, X_test_resnet, y_test)

print(f"ResNet50 embedding + KNN test accuracy: {resnet_acc}")