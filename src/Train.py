import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import get_dataloaders
from Models import Senet50, Senet101

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
NUM_CLASSES = 2
DATA_PATH = "/kaggle/input/breastdm/cls/img9Se"

# ImageNet mean/std
IMAGENET_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
IMAGENET_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)

def build_model(model_name="senet50"):
    if model_name=="senet50":
        model = Senet50(NUM_CLASSES)
    elif model_name=="senet101":
        model = Senet101(NUM_CLASSES)
    else:
        raise ValueError("Only senet50/senet101 supported")
    return model

# Forward 9 slices
def forward_slices(model, x):
    B,S,H,W = x.shape  # (B,9,H,W)
    x = x.view(B*S,1,H,W).repeat(1,3,1,1)  # (B*9,3,H,W)
    x = (x - IMAGENET_MEAN)/IMAGENET_STD
    out = model(x)             # (B*9,2)
    out = out.view(B,S,-1)     # (B,9,2)
    out = out.mean(dim=1)      # average
    return out

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for x,y in tqdm(loader):
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = forward_slices(model,x)
        loss = criterion(outputs,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1)==y).sum().item()
    acc = correct/len(loader.dataset)
    return total_loss/len(loader), acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            outputs = forward_slices(model,x)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            correct += (outputs.argmax(1)==y).sum().item()
    acc = correct/len(loader.dataset)
    return total_loss/len(loader), acc

def main():
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH,BATCH_SIZE)
    print("Building model...")
    model = build_model("senet50").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc>best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),"best_model.pth")
            print("✅ Saved best model!")

    print("\nTesting best model...")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_acc = evaluate(model,test_loader,criterion)
    print(f"Test Acc: {test_acc:.4f}")

if __name__=="__main__":
    main()