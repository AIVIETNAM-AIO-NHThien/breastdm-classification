# config.py
import torch

# ==================== DATA ====================
DATA_PATH = "/kaggle/input/breastdm/cls/img9Se"
BATCH_SIZE = 4
NUM_WORKERS = 2

# ==================== MODEL ====================
NUM_CLASSES = 2
LOAD_VIT = True
VIT_PRETRAINED_PATH = "./model/vit_base_patch16_224_in21k.pth"

# ==================== TRAINING ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 0  # số epoch không cải thiện thì dừng

# ==================== HYPERPARAMETERS ====================
OPTIMIZER = "AdamW"           # "AdamW" hoặc "SGD"
LR = 1e-4
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9 if OPTIMIZER == "SGD" else None

# Learning rate scheduler
SCHEDULER = "ReduceLROnPlateau"   # "ReduceLROnPlateau" hoặc "CosineAnnealing"
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# Regularization (dropout)
DROPOUT = 0.2
ATTN_DROPOUT = 0.1

# ==================== CLASS WEIGHTS ====================
# Nếu muốn tính từ loader, để None; nếu muốn set tay, gán giá trị
CLASS_WEIGHTS = None  # ví dụ [1.0, 1.0] hoặc [0.8, 1.2]

# ==================== SEED ====================
RANDOM_SEED = 42