from pca import *
from cnn import *

import os
import math
import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from visualize import plot_pca_first_two_components, plot_pca_first_three_components

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "data", "filtered_noisy")

# Number of PCs to test (kept for compatibility; CV is still optional)
# Found that running CV would take too long, so take the results from previous 
num_pcs = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

def list_image_paths_and_labels(root_dir, image_exts={".jpg", ".jpeg", ".png"}):
	"""Return (paths, labels) by scanning `root_dir/CLASS/*.png` etc."""
	paths = []
	labels = []

	for label in sorted(os.listdir(root_dir)):
		folder_path = os.path.join(root_dir, label)
		if not os.path.isdir(folder_path):
			continue

		for filename in os.listdir(folder_path):
			ext = os.path.splitext(filename)[1].lower()
			if ext not in image_exts:
				continue
			paths.append(os.path.join(folder_path, filename))
			labels.append(label)

	return paths, labels


def iter_preprocessed_batches(image_paths, batch_size=128):
	"""Yield (X_batch, idx_start, idx_end) for a list of image paths."""
	for idx_start in range(0, len(image_paths), batch_size):
		idx_end = min(idx_start + batch_size, len(image_paths))
		batch_paths = image_paths[idx_start:idx_end]

		images = []
		for path in batch_paths:
			with Image.open(path) as img:
				images.append(img.convert("L").copy())

		X_batch = preprocess_images(images).astype(np.float32)
		yield X_batch, idx_start, idx_end

def iter_image_batches(image_paths, batch_size=64):
    for idx_start in range(0, len(image_paths), batch_size):
        idx_end = min(idx_start + batch_size, len(image_paths))
        batch_paths = image_paths[idx_start:idx_end]

        images = []
        for path in batch_paths:
            with Image.open(path) as img:
                images.append(img.convert("L").copy())

        yield images, idx_start, idx_end


### Load data (paths + labels only)
image_paths, labels = list_image_paths_and_labels(ROOT_DIR)
y = np.asarray(labels)

paths_train, paths_test, y_train, y_test = train_test_split(
	image_paths,
	y,
	test_size=0.2,
	stratify=y,
	random_state=42,
)

print(f"Loaded file list: {len(image_paths)} images")
print(f"Train/test split: {len(paths_train)}/{len(paths_test)}")


### Run IncrementalPCA on training batches
best_p = 200 # Change here if not doing CV
batch_size = 256

ipca = IncrementalPCA(n_components=best_p, batch_size=batch_size)

print("Fitting IncrementalPCA...")

n_fit_batches = math.ceil(len(paths_train) / batch_size)
n_train_batches = n_fit_batches
n_test_batches = math.ceil(len(paths_test) / batch_size)

for X_batch, _, _ in tqdm(
	iter_preprocessed_batches(paths_train, batch_size=batch_size),
	total=n_fit_batches,
	desc="IPCA partial_fit",
	unit="batch",
):
	ipca.partial_fit(X_batch)

print("Transforming train/test into PCA space...")
X_train_pca = np.empty((len(paths_train), best_p), dtype=np.float32)
X_test_pca = np.empty((len(paths_test), best_p), dtype=np.float32)

for X_batch, idx_start, idx_end in tqdm(
	iter_preprocessed_batches(paths_train, batch_size=batch_size),
	total=n_train_batches,
	desc="Transform train",
	unit="batch",
):
	X_train_pca[idx_start:idx_end] = ipca.transform(X_batch).astype(np.float32)

for X_batch, idx_start, idx_end in tqdm(iter_preprocessed_batches(paths_test, batch_size=batch_size), total=n_test_batches):
	X_test_pca[idx_start:idx_end] = ipca.transform(X_batch).astype(np.float32)

print("Projected test data on PCs")


### KNN on PCA features
model = train_knn(X_train_pca, y_train, k=3)
test_acc = evaluate_model(model, X_test_pca, y_test)

print(f"Size of testing dataset: {X_test_pca.shape[0]} by {X_test_pca.shape[1]}")
print(f"Test accuracy under {best_p} principal components: {test_acc}")


### Plots
# plot_pca_first_two_components(X_train_pca, y_train, title="Train set (noisy): PC1 vs PC2")
# plot_pca_first_three_components(X_train_pca, y_train, title="Train set (noisy): PC1 vs PC2 vs PC3")

print("Extracting ResNet50 embeddings...")
resnet = load_resnet50()
transform = get_resnet_transform()

resnet_batch_size = 32 
X_train_resnet = np.empty((len(paths_train), 2048), dtype=np.float32)
X_test_resnet = np.empty((len(paths_test), 2048), dtype=np.float32)

for images_batch, idx_start, idx_end in tqdm(iter_image_batches(paths_train, batch_size=resnet_batch_size), total=math.ceil(len(paths_train) / resnet_batch_size)):
    feats = extract_resnet_features(images_batch, resnet, transform).astype(np.float32)
    X_train_resnet[idx_start:idx_end] = feats

for images_batch, idx_start, idx_end in tqdm(iter_image_batches(paths_test, batch_size=resnet_batch_size), total=math.ceil(len(paths_test) / resnet_batch_size),):
    feats = extract_resnet_features(images_batch, resnet, transform).astype(np.float32)
    X_test_resnet[idx_start:idx_end] = feats

resnet_knn = train_knn(X_train_resnet, y_train, k=3)
resnet_acc = evaluate_model(resnet_knn, X_test_resnet, y_test)
print(f"ResNet50 embedding + KNN test accuracy: {resnet_acc}")