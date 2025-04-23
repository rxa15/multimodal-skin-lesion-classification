import torch
from torchvision.models import resnet50, ResNet50_Weights

# Load pretrained ResNet model
resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Modify the final layer of ResNet to fit your task (e.g., if you're doing binary classification)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, 512)
