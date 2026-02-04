import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

from data.simulate_eeg import SimulatedEEGDataset
from models.spectralMLP import SpectralMLP
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct / total

def main():
    # Set up the device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- DATA PARAMETERS ----------------
    # Define parameters for simulating EEG data and creating the dataset
    num_channels = 16
    total_timesteps = 1024
    window_size = 256
    stride = 128
    num_samples_per_class = 50

    # ---------------- MODEL PARAMETERS ----------------
    # Define the number of output classes for the classification task
    num_classes = 4

    # ---------------- TRAINING PARAMETERS ----------------
    # Define hyper-parameters for the training process
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 30
    early_stop_patience = 5

    # ---------------- DATASET PREPARATION ----------------
    # Create the simulated EEG dataset
    dataset = SimulatedEEGDataset(
        num_samples_per_class=num_samples_per_class,
        num_channels=num_channels,
        total_timesteps=total_timesteps,
        window_size=window_size,
        stride=stride
    )

    # Split the dataset into training and testing sets (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for efficient batching during training and evaluation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get a sample to determine the number of frequency bins for the model's input layer
    x_sample, _ = dataset[0]
    freq_bins = x_sample.shape[1]

    # ---------------- MODEL, LOSS, OPTIMIZER ----------------
    # Initialize the SpectralMLP model and move it to the specified device (GPU/CPU)
    model = SpectralMLP(
        num_channels=num_channels,
        freq_bins=freq_bins,
        num_classes=num_classes
    ).to(device)

    # Define the loss function (Cross Entropy for classification with multiple classes)
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer (Adam is a common choice for deep learning, updates model weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Set up a learning rate scheduler to reduce LR if validation loss plateaus
    # This helps in fine-tuning the model when it's no longer making significant progress
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # ---------------- TRAINING LOOP ----------------
    # Lists to store training and validation metrics (loss and accuracy) for plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Variables for early stopping to prevent overfitting and save computation time
    best_val_acc = 0.0
    epochs_no_improve = 0

    # Iterate over the specified number of epochs (full passes through the training data)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over batches in the training loader
        for x, y in train_loader:
            x = x.to(device) # Move input data (spectral windows) to the selected device
            y = y.to(device) # Move target labels to the selected device

            logits = model(x) # Perform forward pass: model predicts raw scores (logits)
            loss = criterion(logits, y) # Compute loss: measures how far predictions are from true labels

            optimizer.zero_grad() # Clear previous gradients to prevent accumulation
            loss.backward() # Perform backpropagation: computes gradients of loss w.r.t. model parameters
            optimizer.step() # Update model parameters: adjusts weights based on gradients

            preds = logits.argmax(dim=1) # Get predicted class by finding the index of max logit
            correct += (preds == y).sum().item() # Count correct predictions in the current batch
            total += y.size(0) # Accumulate total samples processed
            running_loss += loss.item() # Accumulate loss for the current batch

        # Calculate average training loss and accuracy for the epoch
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Evaluate model performance on the test/validation set (no gradient computation here)
        val_loss, val_acc = evaluate(model, test_loader, device)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss)

        # Record metrics for plotting later
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Print epoch-wise statistics to monitor training progress
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # -------- Early stopping logic --------
        # Check if validation accuracy has improved compared to the best seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc # Update best validation accuracy
            epochs_no_improve = 0 # Reset counter if improved
        else:
            epochs_no_improve += 1 # Increment counter if no improvement

        # Stop training if validation accuracy hasn't improved for 'early_stop_patience' epochs
        if epochs_no_improve >= early_stop_patience:
            print("Early stopping triggered")
            break # Exit the training loop

    # ---------------- FINAL EVALUATION ----------------
    model.eval() # Set model to evaluation mode (disables dropout, uses running averages for batch norm)
    all_preds, all_labels = [], []

    # Perform inference on the entire test set without gradient calculation
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1) # Get the predicted class for each sample

            all_preds.extend(preds.cpu().numpy()) # Store predictions, moving them to CPU
            all_labels.extend(y.cpu().numpy()) # Store true labels, moving them to CPU

    # Compute the confusion matrix and final test accuracy
    cm = confusion_matrix(all_labels, all_preds)
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    print("\nFINAL TEST ACCURACY:", test_acc)
    print("CONFUSION MATRIX:")
    print(cm)

    # ---------------- PLOTTING RESULTS ----------------
    # Create a figure with two subplots for loss and accuracy curves
    plt.figure(figsize=(12, 4))

    # Plot training and validation loss on the first subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")

    # Plot training and validation accuracy on the second subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.show() # Display the plots

# Ensure main() is called only when the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    main()