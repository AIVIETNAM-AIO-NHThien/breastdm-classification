import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from data_loader import get_dataloaders
from Fusion import FusionM_9ch
import config 

# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        
        optimizer.zero_grad()
        
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
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
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
    labels = []
    for _, y in loader:
        labels.extend(y.cpu().numpy())
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.FloatTensor(weights).to(config.DEVICE)


# ==================== MAIN ====================
def main():
    print("="*50)
    print("FUSION MODEL TRAINING (SE-ResNet50 + ViT)")
    print("="*50)
    
    # 1. Load data
    print("\n[1] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.DATA_PATH, config.BATCH_SIZE, config.NUM_WORKERS
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 2. Class weights
    print("\n[2] Computing class weights...")
    if config.CLASS_WEIGHTS is None:
        class_weights = compute_class_weights(train_loader)
    else:
        class_weights = torch.FloatTensor(config.CLASS_WEIGHTS).to(config.DEVICE)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # 3. Build model
    print("\n[3] Building Fusion Model...")
    model = FusionM_9ch(
        num_classes=config.NUM_CLASSES,
        load_vit=config.LOAD_VIT,
        vit_pretrained_path=config.VIT_PRETRAINED_PATH if config.LOAD_VIT else None,
        drop_ratio=config.DROPOUT,
        attn_drop_ratio=config.ATTN_DROPOUT
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 4. Setup optimizer and loss
    print("\n[4] Setting up optimizer...")
    if config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    if config.SCHEDULER == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    else:  # ReduceLROnPlateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler() if config.DEVICE == "cuda" else None
    
    # 5. Training loop with early stopping
    print("\n[5] Starting training...")
    best_val_acc = 0
    best_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config.EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
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
        
        # Scheduler step
        if config.SCHEDULER == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_fusion_model.pth')
            print(f"   ✅ Saved best model! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"   ⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
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
    
    # Save results
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
    
    # Plot curves
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("\n📈 Training curves saved as 'training_curves.png'")
    except:
        pass
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()