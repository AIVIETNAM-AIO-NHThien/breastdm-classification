import os
import torch
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import timm

from data_loader import load_testing
import Models

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../input/roi-classification')
parser.add_argument('--checkpoint_path', type=str, default='Best_Models')
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_loader, test_labels = load_testing(args.data_path, 'test', args.batch_size)
print(f"Test set loaded: {len(test_labels)} images")

# ---------------- Build model from checkpoint filename ----------------
def build_model_from_filename(filename, num_classes=2):
    fname = filename.lower()

    if 'resnet18' in fname: return Models.ResNet18(num_classes)
    if 'resnet50' in fname: return Models.ResNet50(num_classes)
    if 'resnet101' in fname: return Models.ResNet101(num_classes)
    if 'densenet169' in fname: return Models.DenseNet169(num_classes)
    if 'densenet201' in fname: return Models.DenseNet201(num_classes)
    if 'vgg16' in fname: return Models.VGG16(num_classes)
    if 'senet50' in fname: return Models.SENet50(num_classes)
    if 'resnext101' in fname: return Models.ResNeXt101(num_classes)

    if 'vit' in fname:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        return model

    raise ValueError(f"Unknown model type: {filename}")

# ---------------- Load checkpoints ----------------
checkpoint_files = sorted([f for f in os.listdir(args.checkpoint_path) if f.endswith('.pth')])
print(f"Found {len(checkpoint_files)} model(s)")

results = []

for ckpt in checkpoint_files:
    print(f"\n===== Evaluating {ckpt} =====")

    model = build_model_from_filename(ckpt, num_classes=2)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, ckpt), map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_probs = [], []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Testing {ckpt}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(test_labels, all_preds)
    auc = roc_auc_score(test_labels, all_probs)
    print(f"Test Accuracy: {acc:.4f} | Test AUC: {auc:.4f}")

    results.append((ckpt, acc, auc))

# ---------------- Summary ----------------
results.sort(key=lambda x: x[2], reverse=True)

print("\n========== FINAL SUMMARY ==========")
print(f"{'Model':35s} {'Accuracy':>10s} {'AUC':>10s}")
for ckpt, acc, auc in results:
    print(f"{ckpt:35s} {acc:10.4f} {auc:10.4f}")
