import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW
from tqdm import tqdm

from dataset import AnamnesysDataset
from models import AnamnesysModel


# Load the dataset
dataset = AnamnesysDataset("model_input.csv")

# Initialize the model
model = AnamnesysModel()

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# Set the number of epochs and device (GPU or CPU)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and evaluation loop
num_epochs = 3  # You can change this number based on your needs


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct_predictions = 0
    total_predictions = 0

    # Training phase
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

    # Compute training metrics
    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = correct_predictions.double() / total_predictions

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
    )

    # Evaluation phase on validation set
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for inputs, labels in tqdm(
            val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
        ):
            print(type(inputs))
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

    # Compute validation metrics
    val_loss = val_loss / len(val_dataloader)
    val_accuracy = correct_predictions.double() / total_predictions

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
