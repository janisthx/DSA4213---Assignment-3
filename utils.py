import torch
from sklearn.metrics import classification_report

def train(model, dataloader, optimizer, criterion, device="cuda"):
    model.train()
    for epoch in range(3):
        total_loss, correct, total = 0, 0, 0
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} - Loss: {total_loss:.2f} - Accuracy: {acc:.4f}")

def evaluate(model, dataloader, device="cuda"):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1)

            preds_all.extend(preds.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    print("Test Accuracy:", (torch.tensor(preds_all) == torch.tensor(labels_all)).float().mean().item())
    print(classification_report(labels_all, preds_all))