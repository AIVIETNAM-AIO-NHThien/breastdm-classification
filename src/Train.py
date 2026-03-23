import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from data_loader import get_dataloaders
from Fusion import FusionM_9ch

# ==================== CONFIGURATION ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4  
LR = 1e-4  
NUM_CLASSES = 2
DATA_PATH = "/kaggle/input/breastdm/cls/img9Se"
VIT_PRETRAINED_PATH = "./model/vit_base_patch16_224_in21k.pth"  
LOAD_VIT = True  # Set pretrained

# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision training (optional, helps with memory)
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(loader), correct / total


def compute_class_weights(loader):
    """Compute class weights for imbalanced dataset"""
    labels = []
    for _, y in loader:
        labels.extend(y.cpu().numpy())
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.FloatTensor(weights).to(DEVICE)


# ==================== MAIN ====================
def main():
    print("="*50)
    print("FUSION MODEL TRAINING (SE-ResNet50 + ViT)")
    print("="*50)
    
    # 1. Load data
    print("\n[1] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_PATH, BATCH_SIZE)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Compute class weights (optional)
    print("\n[2] Computing class weights...")
    class_weights = compute_class_weights(train_loader)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # 3. Build model
    print("\n[3] Building Fusion Model...")
    model = FusionM_9ch(
        num_classes=NUM_CLASSES,
        load_vit=LOAD_VIT,
        vit_pretrained_path=VIT_PRETRAINED_PATH if LOAD_VIT else None
    )
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Setup optimizer and loss
    print("\n[4] Setting up optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Mixed precision training (if CUDA available)
    scaler = torch.cuda.amp.GradScaler() if DEVICE == "cuda" else None
    
    # 5. Training loop
    print("\n[5] Starting training...")
    best_val_acc = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update scheduler
        scheduler.step()
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_fusion_model.pth')
            print(f"   ✅ Saved best model! (Val Acc: {val_acc:.4f})")
    
    # 6. Test best model
    print(f"\n{'='*50}")
    print("TESTING BEST MODEL")
    print(f"{'='*50}")
    
    checkpoint = torch.load('best_fusion_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    print(f"\n🎯 Final Results:")
    print(f"   Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"   Test Acc: {test_acc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    torch.save(results, 'training_results.pth')
    
    # Plot training curves (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        print("\n📈 Training curves saved as 'training_curves.png'")
    except:
        pass
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()