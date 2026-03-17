import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import get_dataloaders
import timm  # pretrained model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
NUM_CLASSES = 2

DATA_PATH = "/kaggle/input/breastdm/cls/img9Se"

def build_model():
    model = timm.create_model(
        "seresnet50", 
        pretrained=True, 
        num_classes=NUM_CLASSES
    )
    return model

# FORWARD 9 SLICES
def forward_slices(model, x):

    outputs = []

    for i in range(9):
        slice_i = x[:, i, :, :]          # (B,H,W)
        slice_i = slice_i.unsqueeze(1)   # (B,1,H,W)
        slice_i = slice_i.repeat(1,3,1,1) # (B,3,H,W)

        out = model(slice_i)             # (B,2)
        outputs.append(out)

    outputs = torch.stack(outputs, dim=0)  # (9,B,2)
    outputs = outputs.mean(0)              # (B,2)

    return outputs


# TRAIN
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0

    for x, y in tqdm(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = forward_slices(model, x)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(1)
        correct += (pred == y).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# VALIDATE
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            outputs = forward_slices(model, x)

            loss = criterion(outputs, y)
            total_loss += loss.item()

            pred = outputs.argmax(1)
            correct += (pred == y).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# MAIN
def main():
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        DATA_PATH, BATCH_SIZE
    )

    print("Building model...")
    model = build_model().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved best model!")

    print("\nTesting best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()