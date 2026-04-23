import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image

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