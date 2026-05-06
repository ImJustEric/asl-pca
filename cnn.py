import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_resnet50():
    """Load pretrained ResNet50 with final classification layer removed so we have the embeddings"""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

def get_resnet_transform():
    """Default ResNet50 transformation to process inputs"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_resnet_features(images, model, transform):
    """
    Extract ResNet50 features for a list/array of grayscale images.
    Returns feature matrix of shape (n_samples, 2048)
    """
    features = []

    for img in images:
        img = Image.fromarray(np.array(img).astype(np.uint8))
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            feat = model(img_tensor)

        feat = feat.view(-1).numpy()
        features.append(feat)

    return np.array(features)

def plot_resnet_tsne(embeddings, labels=None, class_names=None, perplexity=30, random_state=42):
    """
    Plot 2D t-SNE projection of CNN embeddings
    """
    embeddings = np.asarray(embeddings)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    projected = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 6))
    if labels is None:
        ax.scatter(projected[:, 0], projected[:, 1], s=20, alpha=0.7)
    else:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            name = class_names[label] if class_names is not None else str(label)
            ax.scatter(projected[mask, 0], projected[mask, 1], s=20, alpha=0.7, label=name)
        ax.legend(
            title="Class",
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize="small",
            ncols=2,
        )

    ax.set_title("t-SNE of CNN Embeddings")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    plt.show()