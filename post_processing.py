import matplotlib.pyplot as plt
import json

def plot_losses(train_losses, val_losses, test_loss):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")

    plt.axhline(y=test_loss, color="r", linestyle="--", label="Test Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training, Validation, and Test Loss")
    plt.legend()
    plt.grid()
    plt.show()


def save_losses(train_losses, val_losses, test_loss, filename="losses.json"):
    losses = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
    }
    with open(filename, "w") as file:
        json.dump(losses, file)
    print(f"Losses saved to {filename}")