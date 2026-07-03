

import torch
from sklearn.metrics import classification_report, f1_score


def evaluate_model(model, test_loader, loss_func, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "f1": f1,
        "all_preds": all_preds,
        "all_labels": all_labels,
    }
