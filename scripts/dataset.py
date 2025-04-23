from transformers import MPNetTokenizer
from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import os


class AnamnesysDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        # Load the dataset
        self.data = pd.read_csv(file_path)
        self.tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
        self.max_length = max_length

        # Map labels to integers
        self.label_map = {
            label: idx for idx, label in enumerate(self.data["class"].unique())
        }
        self.labels = self.data["class"].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input text and label
        text = self.data.iloc[idx]["input_text"]
        label = self.labels.iloc[idx]

        # Tokenize the input text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Extract input_ids and attention_mask
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SkinImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = [], []
        self.label_to_idx = {"DA": 0, "Non-DA": 1}  # Assign class indices explicitly

        for label in ["DA", "Non-DA"]:  # Iterate through fixed class names
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
