import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


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


def main():

    # Load processed dataset
    X = np.load("Data/Processed/X.npy")
    y = np.load("Data/Processed/y.npy")

    # Same split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

    # Rebuild model
    model = SpectraCNN(input_length=X.shape[1])

    # Load trained weights
    model.load_state_dict(torch.load("models/cnn_model.pth", map_location="cpu"))

    model.eval()

    # FAST inference (single forward pass)
    with torch.no_grad():

        outputs = model(X_test)

        preds = torch.argmax(outputs, dim=1).numpy()

    accuracy = (preds == y_test).mean()

    print("\nSaved CNN Model Accuracy:", accuracy)

    label_names = ["Hot", "Medium", "Cool"]

    print("\nClassification Report:\n")

    print(classification_report(y_test, preds, target_names=label_names))

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)

    disp.plot(cmap="viridis")

    plt.title("Saved CNN Confusion Matrix")

    plt.show()


if __name__ == "__main__":
    main()