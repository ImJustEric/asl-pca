"""Visualization helpers for the ASL PCA/CNN experiments."""
import numpy as np
import matplotlib.pyplot as plt


def plot_pca_first_two_components(X_pca, labels, title="PCA projection (PC1 vs PC2)", save_path=None):
	"""Scatter-plot points on the first two principal components.
	Uses the PCA results as inputs, along with their respective labels
	"""

	# Internal style defaults (change if we want different)
	figsize = (10, 8)
	point_size = 10.0
	alpha = 0.7

	X_pca = np.asarray(X_pca)
	if X_pca.ndim != 2 or X_pca.shape[1] < 2:
		raise ValueError(f"X_pca must have shape (n_samples, n_components>=2); got {X_pca.shape}")

	labels_arr = np.asarray(list(labels))
	if labels_arr.shape[0] != X_pca.shape[0]:
		raise ValueError(f"labels length ({labels_arr.shape[0]}) must match X_pca rows ({X_pca.shape[0]})")

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
