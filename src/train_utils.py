

import matplotlib.pyplot as plt
import torch
import random
import numpy as np


def train_model(model, train_loader, val_loader, optimizer, loss_func, device, num_epochs):


    training_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        model.eval()
        val_loss = 0.0
        wrong = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_func(outputs, labels)

                predictions = torch.argmax(outputs, dim=1)
                wrong += (predictions != labels).sum().item()
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_loss_avg = training_loss / len(train_loader)
        accuracy = 1 - wrong / len(val_loader.dataset)

        print(
            f"Epoha {epoch + 1}/{num_epochs} | "
            f"Train loss: {train_loss_avg:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        training_losses.append(train_loss_avg)
        val_losses.append(val_loss)

    return training_losses, val_losses


def plot_losses(training_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoha")
    plt.ylabel("Loss")
    plt.title("Kriva gubitka kroz epohe treninga")
    plt.legend()
    plt.show()
    
def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
