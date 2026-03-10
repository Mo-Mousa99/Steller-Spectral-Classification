import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset


# Load dataset
X = np.load("Data/Processed/X.npy")
y = np.load("Data/Processed/y.npy")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test_tensor),
    batch_size=32,
    shuffle=False
)


class SpectraCNN(nn.Module):

    def __init__(self, input_length):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            conv_out = self.conv(dummy)
            self.flatten_size = conv_out.shape[1] * conv_out.shape[2]

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Initialize model
model = SpectraCNN(input_length=X.shape[1])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training loop
for epoch in range(10):

    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:

        optimizer.zero_grad()

        preds = model(xb)

        loss = criterion(preds, yb)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


# Evaluation
model.eval()

all_preds = []
all_true = []

with torch.no_grad():

    for xb, yb in test_loader:

        outputs = model(xb)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.numpy())
        all_true.extend(yb.numpy())


all_preds = np.array(all_preds)
all_true = np.array(all_true)

accuracy = (all_preds == all_true).mean()

print("\nTest accuracy:", accuracy)

label_names = ["Hot", "Medium", "Cool"]

print("\nClassification Report:\n")
print(classification_report(all_true, all_preds, target_names=label_names))


# Confusion matrix
cm = confusion_matrix(all_true, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_names
)

disp.plot(cmap="viridis")

plt.title("CNN Confusion Matrix")

plt.show()


# Save trained model
os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/cnn_model.pth")

print("\nModel saved to models/cnn_model.pth")