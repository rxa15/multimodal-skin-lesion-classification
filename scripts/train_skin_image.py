import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataset import SkinImageDataset
from models import SkinImageModel


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize to the input size of ResNet50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

root_dir = "dataset/processed"

dataset = SkinImageDataset(root_dir=root_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = SkinImageModel(num_classes=2).to(device)  # Use the appropriate device (GPU/CPU)

criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

num_epochs = 100  # Set number of epochs for training

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", unit="batch"
    ) as tepoch:
        for inputs, labels in tepoch:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)

            # Get accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            # Update the progress bar description
            tepoch.set_postfix(
                loss=loss.item(),
                accuracy=correct_predictions.double() / total_predictions,
            )

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions.double() / total_predictions

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}"
    )

    # Step 6: Validation Loop with tqdm for Progress Bar
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0

    # Create a tqdm progress bar for validation
    with tqdm(
        val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", unit="batch"
    ) as tepoch:
        with torch.no_grad():  # No need to calculate gradients during validation
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Update validation loss
                val_loss += loss.item() * inputs.size(0)

                # Get accuracy
                _, preds = torch.max(outputs, 1)
                val_correct_predictions += torch.sum(preds == labels.data)
                val_total_predictions += labels.size(0)

                # Update the progress bar description
                tepoch.set_postfix(
                    loss=loss.item(),
                    accuracy=val_correct_predictions.double() / val_total_predictions,
                )

    # Calculate average validation loss and accuracy
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct_predictions.double() / val_total_predictions

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


# Step 7: Save the Model
torch.save(model.state_dict(), "skin_image_model.pth")
print("Model saved to 'skin_image_model.pth'")
