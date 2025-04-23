from dataset import AnamnesysDataset
from models import MultimodalModel
import torch

text_input = "Hello, how are you doing today?"
image_input = torch.randn(1, 3, 224, 224)
model = MultimodalModel()

outputs = model(image_input, text_input)

print(outputs)

# # Get the last hidden states (shape: [batch_size, seq_len, hidden_size])
# last_hidden_state = outputs.last_hidden_state

# # Get the pooler output (usually used for classification tasks) (shape: [batch_size, hidden_size])
# pooler_output = outputs.pooler_output

# # Print the shapes of the outputs
# print("Last hidden state shape:", last_hidden_state.shape)
# print("Pooler output shape:", pooler_output.shape)
