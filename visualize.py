"""Visualization helpers for the ASL PCA/CNN experiments."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


def plot_pca_first_two_components(X_pca, labels, title="PCA projection (PC1 vs PC2)", save_path=None):
	"""Scatter-plot points on the first two principal components.
	Uses the PCA results as inputs, along with their respective labels
	"""

	# Internal style defaults (change if we want different)
	figsize = (10, 8)
	point_size = 10.0
	alpha = 0.7

	X_pca = np.asarray(X_pca)
	labels_arr = np.asarray(list(labels))
	unique_labels = np.unique(labels_arr.astype(str))
	num_classes = unique_labels.shape[0]

	fig, ax = plt.subplots(figsize=figsize)

	cmap_name = "tab20" if num_classes <= 20 else "gist_ncar"
	cmap = plt.get_cmap(cmap_name, num_classes)

	for idx, label in enumerate(unique_labels):
		mask = labels_arr.astype(str) == label
		ax.scatter(X_pca[mask, 0],
			X_pca[mask, 1],
			s=point_size,
			alpha=alpha,
			color=cmap(idx),
			label=label,
			edgecolors="none",
		)

	ax.set_title(title)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")

	ax.legend(title="Class", bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize="small", ncols=2,)

	fig.tight_layout()

	if save_path is not None:
		fig.savefig(save_path, bbox_inches="tight", dpi=300)

	plt.show()

	return ax


def plot_pca_first_three_components(X_pca, labels, title="PCA projection (PC1 vs PC2 vs PC3)", save_path=None):
	"""Scatter-plot points on the first three principal components.
	Uses the PCA results as inputs, along with their respective labels.
	"""

	# Internal style defaults (match the 2D helper)
	figsize = (10, 8)
	point_size = 10.0
	alpha = 0.7

	X_pca = np.asarray(X_pca)

	labels_arr = np.asarray(list(labels))
	unique_labels = np.unique(labels_arr.astype(str))
	num_classes = unique_labels.shape[0]

	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111, projection="3d")

	cmap_name = "tab20" if num_classes <= 20 else "gist_ncar"
	cmap = plt.get_cmap(cmap_name, num_classes)

	for idx, label in enumerate(unique_labels):
		mask = labels_arr.astype(str) == label
		ax.scatter(
			X_pca[mask, 0],
			X_pca[mask, 1],
			X_pca[mask, 2],
			s=point_size,
			alpha=alpha,
			color=cmap(idx),
			label=label,
			edgecolors="none",
		)

	ax.set_title(title)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")

	ax.legend(title="Class", bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0, fontsize="small", ncols=2,)

	fig.tight_layout()

	if save_path is not None:
		fig.savefig(save_path, bbox_inches="tight", dpi=300)

	plt.show()

	return ax



def plot_pca_component_images(pca, n_components=10, image_shape=(400, 400), save_path=None):
    """
    Visualize the first n PCA components as 'eigen-images'.
    """
    comps = np.asarray(pca.components_)
    n = min(n_components, comps.shape[0])
    H, W = image_shape

    rows = int(np.ceil(n / 5))
    cols = min(5, n)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n):
        comp_img = comps[i].reshape(H, W)

        vmax = np.max(np.abs(comp_img)) + 1e-12
        axes[i].imshow(comp_img, cmap="gray", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"PC{i+1}")
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"First {n} PCA components (as images)", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return axes[:n]