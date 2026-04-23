import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import os


def load_images_and_labels(root_dir):
    """Given directory is:
    root_dir/
        A/
        B/
        C/
        ...
    Returns images as list of image arrays and labels as list of corresponding folder names
    """
    images = []
    labels = []

    for label in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, label)

        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            try:
                img = Image.open(file_path)
                img = np.array(img)
                images.append(img)
                labels.append(label)
            except:
                continue

    return images, labels

def preprocess_single_image(image_path):
    """
    Load a single image from path and preprocess it to match the training pipeline.
    This will be used for testing single images, rather than 'preprocess_images', which
    takes arrays of images and returns as an array
    """
    # Load image
    img = Image.open(image_path)
    img = np.array(img)
    img = img / 255.0
    img_flat = img.flatten()

    return img_flat

def preprocess_images(images):
    """Take grayscale images and normalize/vectorize"""
    processed = []

    for img in images:
        # Convert to numpy array and normalize
        # Note that images are 400x400
        if isinstance(img, Image.Image):
            if img.size != (400, 400):
                img = img.resize((400, 400))
            img = np.array(img)
        else:
            if img.shape[:2] != (400, 400):
                img = Image.fromarray(img).resize((400, 400))
                img = np.array(img)

        img = img / 255.0
        img_flat = img.flatten()
        processed.append(img_flat)

    return np.array(processed)

def perform_pca(data, num_pcs):
    """Given a data matrix and the number of principal components, perform PCA
    Return: PCA object"""
    return PCA(num_pcs).fit(data)


def project_data(pca, data):
    """Project data into the PCA subspace"""
    return pca.transform(data)


def cross_validate_p(X_train, y_train, num_pcs, k=3, n_splits=5):
    """Perform k-fold cross validation to check which value of num_pc does best"""
    """
    Args:
        X_train, y_train: TRAINING dataset
        num_pcs: The number of principal components we are trying
        k: The number of nearest neighbors
        n_splits: The number of folds to perform
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}

    for num_pc in num_pcs:
        fold_accuracies = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

            # Fit PCA only on fold training data
            pca = perform_pca(X_fold_train, num_pc)

            X_fold_train_pca = project_data(pca, X_fold_train)
            X_fold_val_pca = project_data(pca, X_fold_val)

            model = train_knn(X_fold_train_pca, y_fold_train, k)
            acc = evaluate_model(model, X_fold_val_pca, y_fold_val)

            fold_accuracies.append(acc)

        results[num_pc] = np.mean(fold_accuracies)

    return results

def train_knn(X_train, y_train, k=3):
    """Train a k-NN classifier on projected data"""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Return classification accuracy on a dataset"""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
