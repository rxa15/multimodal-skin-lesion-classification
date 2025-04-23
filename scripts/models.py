from torchvision.models import resnet50, ResNet50_Weights
from transformers import MPNetModel, MPNetTokenizer
import torch.nn as nn
import torch


class MultimodalModel(nn.Module):
    def __init__(self, model_name="microsoft/mpnet-base", num_classes=2):
        super(MultimodalModel, self).__init__()
        self.image_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 768)
        self.text_model = MPNetModel.from_pretrained(model_name)
        self.tokenizer = MPNetTokenizer.from_pretrained(model_name)
        self.classifier = nn.Linear(
            self.text_model.config.hidden_size + 768,
            num_classes,
        )

    def forward(self, image_input, text_input):
        image_features = self.image_model(image_input)
        text_input = self.tokenizer(text_input, return_tensors="pt")
        text_features = self.text_model(**text_input).pooler_output

        combined_features = torch.cat((image_features, text_features), dim=1)
        outputs = self.classifier(combined_features)

        return outputs


class AnamnesysModel(nn.Module):
    def __init__(self, model_name="microsoft/mpnet-base", num_classes=2):
        super(AnamnesysModel, self).__init__()
        self.mpnet_model = MPNetModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.mpnet_model.config.hidden_size, 512),  # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout layer to prevent overfitting
            nn.Linear(
                512, num_classes
            ),  # Output layer (num_classes is the number of classes for classification)
        )

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            output = self.mpnet_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        else:
            output = self.mpnet_model(input_ids=input_ids)

        # Get the hidden state for the [CLS] token (first token) for classification
        # The output of the MPNet model contains several things, but we are interested in `last_hidden_state`
        # The shape of `last_hidden_state` is (batch_size, sequence_length, hidden_size)
        hidden_state = output.last_hidden_state[
            :, 0, :
        ]  # Extract the [CLS] token embedding

        logits = self.classifier(hidden_state)

        return logits


class SkinImageModel(nn.Module):
    def __init__(self, model_name="resnet50", num_classes=2, hidden_size=512):
        super(SkinImageModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, hidden_size)
        self.downstream = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.model(x)
        return self.downstream(x)
