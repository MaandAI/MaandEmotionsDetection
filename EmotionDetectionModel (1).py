

# # ─────────────────────────────────────────────
# # STEP 1 — Imports & Device Setup
# # ─────────────────────────────────────────────
# import os
# import shutil
# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# from PIL import Image

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)


# # ─────────────────────────────────────────────
# # STEP 2 — Paths
# # ─────────────────────────────────────────────
# BASE_DIR    = os.path.join("content", "processed_data")   # your existing folder
# TRAIN_DIR   = os.path.join("content", "dataset", "train")
# TEST_DIR    = os.path.join("content", "dataset", "test")
# MODEL_PATH  = "emotion_model.pth"
# ONNX_PATH   = "emotion_model.onnx"

# CLASSES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# NUM_CLASSES = len(CLASSES)
# IMG_SIZE    = 224
# BATCH_SIZE  = 32
# EPOCHS      = 5


# # ─────────────────────────────────────────────
# # STEP 3 — Train / Test Split  (80 / 20)
# # ─────────────────────────────────────────────
# def split_dataset(base_dir, train_dir, test_dir, split_ratio=0.8):
#     """Split processed_data into train/test folders."""
#     if os.path.exists(train_dir) and os.path.exists(test_dir):
#         print("Dataset already split — skipping.")
#         return

#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir,  exist_ok=True)

#     for cls in os.listdir(base_dir):
#         class_path = os.path.join(base_dir, cls)
#         if not os.path.isdir(class_path):
#             continue

#         images = os.listdir(class_path)
#         random.shuffle(images)
#         split  = int(len(images) * split_ratio)

#         for subset, imgs in [("train", images[:split]), ("test", images[split:])]:
#             dest = os.path.join("content", "dataset", subset, cls)
#             os.makedirs(dest, exist_ok=True)
#             for img in imgs:
#                 shutil.copy(os.path.join(class_path, img),
#                             os.path.join(dest, img))

#     print("Dataset split completed.")
#     print("Train folders:", os.listdir(train_dir))
#     print("Test  folders:", os.listdir(test_dir))


# # ─────────────────────────────────────────────
# # STEP 4 — DataLoaders
# # ─────────────────────────────────────────────
# def get_dataloaders(train_dir, test_dir, img_size, batch_size):
#     train_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ])
#     test_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#     ])

#     train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
#     test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

#     print("Classes:", train_dataset.classes)
#     print("Train samples:", len(train_dataset))
#     print("Test  samples:", len(test_dataset))

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader


# # ─────────────────────────────────────────────
# # STEP 5 — Build Model (MobileNetV3 Transfer Learning)
# # ─────────────────────────────────────────────
# def build_model(num_classes, device):
#     model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
#     model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
#     return model.to(device)


# # ─────────────────────────────────────────────
# # STEP 6 — Training Loop
# # ─────────────────────────────────────────────
# def train(model, train_loader, criterion, optimizer, device, epochs):
#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0

#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss    = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")


# # ─────────────────────────────────────────────
# # STEP 7 — Validation
# # ─────────────────────────────────────────────
# def evaluate(model, test_loader, device):
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs        = model(images)
#             _, predicted   = torch.max(outputs, 1)
#             total         += labels.size(0)
#             correct       += (predicted == labels).sum().item()

#     acc = 100.0 * correct / total
#     print(f"Validation Accuracy: {acc:.2f}%")
#     return acc


# # ─────────────────────────────────────────────
# # STEP 8 — Export to ONNX
# # ─────────────────────────────────────────────
# def export_onnx(model, device, onnx_path):
#     model.eval()
#     dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=11,
#         export_params=True,
#         do_constant_folding=True,
#         training=torch.onnx.TrainingMode.EVAL,
#     )

#     # Remove split data file if created
#     split_file = onnx_path + ".data"
#     if os.path.exists(split_file):
#         os.remove(split_file)

#     print(f"ONNX model saved → {onnx_path}")


# # ─────────────────────────────────────────────
# # STEP 9 — Single Image Prediction
# # ─────────────────────────────────────────────
# def predict_image(model, img_path, classes, device):
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#     ])

#     image = Image.open(img_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     model.eval()
#     with torch.no_grad():
#         output = model(image)
#         pred   = torch.argmax(output, dim=1)

#     print(f"Predicted Emotion: {classes[pred.item()]}")
#     return classes[pred.item()]


# # ─────────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────────
# if __name__ == "__main__":

#     # 1. Split dataset
#     split_dataset(BASE_DIR, TRAIN_DIR, TEST_DIR)

#     # 2. Load data
#     train_loader, test_loader = get_dataloaders(
#         TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
#     )

#     # 3. Build model
#     model     = build_model(NUM_CLASSES, device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # 4. Train
#     train(model, train_loader, criterion, optimizer, device, EPOCHS)

#     # 5. Evaluate
#     evaluate(model, test_loader, device)

#     # 6. Save PyTorch weights
#     torch.save(model.state_dict(), MODEL_PATH)
#     print(f"Model saved → {MODEL_PATH}")

#     # 7. Export ONNX
#     export_onnx(model, device, ONNX_PATH)

#     # 8. Quick prediction test — pick first image from test/happy
#     test_happy = os.path.join(TEST_DIR, "happy")
#     if os.path.exists(test_happy):
#         sample_img = os.path.join(test_happy, os.listdir(test_happy)[0])
#         predict_image(model, sample_img, CLASSES, device)


























# ─────────────────────────────────────────────
# STEP 1 — Imports & Device Setup
# ─────────────────────────────────────────────
import os
import shutil
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Check and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Hardware Optimization: Enable CUDNN auto-tuner for optimal architecture performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("CUDA Optimizations Enabled (CUDNN Benchmark).")


# ─────────────────────────────────────────────
# STEP 2 — Paths & Hyperparameters
# ─────────────────────────────────────────────
BASE_DIR    = os.path.join("content", "processed_data")   
TRAIN_DIR   = os.path.join("content", "dataset", "train")
TEST_DIR    = os.path.join("content", "dataset", "test")
MODEL_PATH  = "emotion_model.pth"
ONNX_PATH   = "emotion_model.onnx"

CLASSES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
BATCH_SIZE  = 32  # Increased from 32 to 64 to leverage your 12GB VRAM on the 4070 Super
EPOCHS      = 10  # Sufficient epochs to see convergence with metrics tracking


# ─────────────────────────────────────────────
# STEP 3 — Train / Test Split (80 / 20)
# ─────────────────────────────────────────────
def split_dataset(base_dir, train_dir, test_dir, split_ratio=0.8):
    """Split processed_data into train/test folders if not already done."""
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("Dataset already split — skipping.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    for cls in os.listdir(base_dir):
        class_path = os.path.join(base_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)
        split  = int(len(images) * split_ratio)

        for subset, imgs in [("train", images[:split]), ("test", images[split:])]:
            dest = os.path.join("content", "dataset", subset, cls)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

    print("Dataset split completed.")


# ─────────────────────────────────────────────
# STEP 4 — High-Speed DataLoaders with Augmentation
# ─────────────────────────────────────────────
def get_dataloaders(train_dir, test_dir, img_size, batch_size):
    # Enhanced training transformations to combat overfitting
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    print("Classes:", train_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Test  samples:", len(test_dataset))

    # Optimization for Ryzen 7 5700X: Utilizing 4 worker threads for pipeline prefetching
    num_workers = 4 
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ─────────────────────────────────────────────
# STEP 5 — Build Model (MobileNetV3 Transfer Learning)
# ─────────────────────────────────────────────
def build_model(num_classes, device):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model.to(device)


# ─────────────────────────────────────────────
# STEP 6 — AMP-Accelerated Training & Evaluation Loop
# ─────────────────────────────────────────────
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    # Initialize Gradient Scaler for FP16 Mixed Precision
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            # non_blocking=True utilizes pinned memory to overlap transfer with CPU tasks
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # Performance tip: set_to_none saves memory bandwidth
            
            # Run forward pass with automatic mixed precision (AMP)
            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            # Scales the loss and performs backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = 100.0 * correct_train / total_train

        # --- VALIDATION ---
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted   = torch.max(outputs, 1)
                total_val     += labels.size(0)
                correct_val   += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        epoch_val_acc  = 100.0 * correct_val / total_val

        # Record metrics
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    return history


# ─────────────────────────────────────────────
# STEP 7 — Plotting Metrics Graph
# ─────────────────────────────────────────────
def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # 1. Loss Figure
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], 'b-o', label='Training Loss')
    plt.plot(epochs, history["val_loss"], 'r-o', label='Validation Loss')
    plt.title('Loss Trends (Train vs Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy Figure
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], 'b-o', label='Training Accuracy')
    plt.plot(epochs, history["val_acc"], 'r-o', label='Validation Accuracy')
    plt.title('Accuracy Trends (Train vs Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("emotion_training_metrics.png")
    plt.show()
    print("Metric graphs saved successfully as 'emotion_training_metrics.png'")


# ─────────────────────────────────────────────
# STEP 8 — Export to ONNX
# ─────────────────────────────────────────────
def export_onnx(model, device, onnx_path):
    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    if os.path.exists(onnx_path + ".data"):
        os.remove(onnx_path + ".data")

    print(f"ONNX model saved → {onnx_path}")


# ─────────────────────────────────────────────
# STEP 9 — Single Image Prediction Test
# ─────────────────────────────────────────────
def predict_image(model, img_path, classes, device):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            output = model(image)
        pred = torch.argmax(output, dim=1)

    print(f"Test Inference Prediction: {classes[pred.item()]}")
    return classes[pred.item()]


# ─────────────────────────────────────────────
# MAIN EXECUTION BLOCK
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Split dataset
    split_dataset(BASE_DIR, TRAIN_DIR, TEST_DIR)

    # 2. Get high-speed data loaders
    train_loader, test_loader = get_dataloaders(
        TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
    )

    # 3. Initialize model components
    model     = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()
    
    # 0.00005 is ideal for fine-tuning pre-trained networks safely without shattering weights
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # 4. Train, track, and evaluate model
    print("\n--- Starting Training Loop ---")
    history = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)

    # 5. Plot and save metric visualization charts
    plot_metrics(history)

    # 6. Save native PyTorch model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # 7. Export runtime optimized graph via ONNX
    export_onnx(model, device, ONNX_PATH)

    # 8. Single image structural verification pass
    test_happy = os.path.join(TEST_DIR, "happy")
    if os.path.exists(test_happy) and len(os.listdir(test_happy)) > 0:
        sample_img = os.path.join(test_happy, os.listdir(test_happy)[0])
        predict_image(model, sample_img, CLASSES, device)





















# # ─────────────────────────────────────────────
# # STEP 1 — Imports & Device Setup
# # ─────────────────────────────────────────────
# import os
# import shutil
# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     print("CUDA Optimizations Enabled (CUDNN Benchmark).")


# # ─────────────────────────────────────────────
# # STEP 2 — Paths & Hyperparameters
# # ─────────────────────────────────────────────
# BASE_DIR    = os.path.join("content", "processed_data")   
# TRAIN_DIR   = os.path.join("content", "dataset", "train")
# TEST_DIR    = os.path.join("content", "dataset", "test")
# MODEL_PATH  = "emotion_model.pth"
# ONNX_PATH   = "emotion_model.onnx"

# CLASSES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# NUM_CLASSES = len(CLASSES)
# IMG_SIZE    = 224
# BATCH_SIZE  = 64  # Optimized for RTX 4070 Super (12GB VRAM)
# EPOCHS      = 15  


# # ─────────────────────────────────────────────
# # STEP 3 — Train / Test Split (80 / 20)
# # ─────────────────────────────────────────────
# def split_dataset(base_dir, train_dir, test_dir, split_ratio=0.8):
#     if os.path.exists(train_dir) and os.path.exists(test_dir):
#         print("Dataset already split — skipping.")
#         return

#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir,  exist_ok=True)

#     for cls in os.listdir(base_dir):
#         class_path = os.path.join(base_dir, cls)
#         if not os.path.isdir(class_path):
#             continue

#         images = os.listdir(class_path)
#         random.shuffle(images)
#         split  = int(len(images) * split_ratio)

#         for subset, imgs in [("train", images[:split]), ("test", images[split:])]:
#             dest = os.path.join("content", "dataset", subset, cls)
#             os.makedirs(dest, exist_ok=True)
#             for img in imgs:
#                 shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

#     print("Dataset split completed.")


# # ─────────────────────────────────────────────
# # STEP 4 — High-Speed DataLoaders (Aggressive Augmentation)
# # ─────────────────────────────────────────────
# def get_dataloaders(train_dir, test_dir, img_size, batch_size):
#     # ANTI-OVERFITTING FIX: Drastically harder augmentations to disrupt memorization
#     train_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(30),                          # Increased to 30 degrees
#         transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)), # Random zooming
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Horizontal/vertical shifting
#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # Heavier lighting shifts
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
    
#     test_transform = transforms.Compose([
#         transforms.Resize((img_size, img_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
#     test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

#     # Multi-threaded pipeline matching your Ryzen 7 5700X
#     num_workers = 4 
    
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=True, prefetch_factor=2
#     )
#     test_loader  = DataLoader(
#         test_dataset,  batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )

#     return train_loader, test_loader


# # ─────────────────────────────────────────────
# # STEP 5 — Build Model (Freezing the Backbone)
# # ─────────────────────────────────────────────
# def build_model(num_classes, device):
#     # Load pretrained MobileNetV3 Large
#     model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
#     # ANTI-OVERFITTING FIX: Freeze feature extraction layers so they cannot be modified/overwritten
#     for param in model.parameters():
#         param.requires_grad = False
        
#     # Replace final linear layer (This automatically defaults to requires_grad=True)
#     model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
#     return model.to(device)


# # ─────────────────────────────────────────────
# # STEP 6 — Loop with Automatic Checkpointing
# # ─────────────────────────────────────────────
# def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs, save_path):
#     history = {
#         "train_loss": [], "train_acc": [],
#         "val_loss": [], "val_acc": []
#     }

#     best_val_loss = float('inf') 
#     scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

#     for epoch in range(epochs):
#         # --- TRAINING ---
#         model.train()
#         running_loss, correct_train, total_train = 0.0, 0, 0

#         for images, labels in train_loader:
#             images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
            
#             with torch.amp.autocast(device_type=device.type):
#                 outputs = model(images)
#                 loss    = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()

#         epoch_train_loss = running_loss / len(train_loader.dataset)
#         epoch_train_acc  = 100.0 * correct_train / total_train

#         # --- VALIDATION ---
#         model.eval()
#         running_val_loss, correct_val, total_val = 0.0, 0, 0

#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
#                 with torch.amp.autocast(device_type=device.type):
#                     outputs = model(images)
#                     loss    = criterion(outputs, labels)

#                 running_val_loss += loss.item() * images.size(0)
#                 _, predicted   = torch.max(outputs, 1)
#                 total_val     += labels.size(0)
#                 correct_val   += (predicted == labels).sum().item()

#         epoch_val_loss = running_val_loss / len(test_loader.dataset)
#         epoch_val_acc  = 100.0 * correct_val / total_val

#         # Record metrics
#         history["train_loss"].append(epoch_train_loss)
#         history["train_acc"].append(epoch_train_acc)
#         history["val_loss"].append(epoch_val_loss)
#         history["val_acc"].append(epoch_val_acc)

#         status_msg = (f"Epoch [{epoch+1}/{epochs}] "
#                       f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
#                       f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
#         # Real-time Checkpointing
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(model.state_dict(), save_path)
#             status_msg += "  --> [SAVED BEST MODEL!]"

#         print(status_msg)

#     return history


# # ─────────────────────────────────────────────
# # STEP 7 — Plotting Metrics Graph
# # ─────────────────────────────────────────────
# def plot_metrics(history):
#     epochs = range(1, len(history["train_loss"]) + 1)
#     plt.figure(figsize=(14, 5))

#     # Loss Figure
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history["train_loss"], 'b-o', label='Training Loss')
#     plt.plot(epochs, history["val_loss"], 'r-o', label='Validation Loss')
#     plt.title('Loss Trends (Train vs Val)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # Accuracy Figure
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history["train_acc"], 'b-o', label='Training Accuracy')
#     plt.plot(epochs, history["val_acc"], 'r-o', label='Validation Accuracy')
#     plt.title('Accuracy Trends (Train vs Val)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig("emotion_training_metrics.png")
#     plt.show()
#     print("Performance charts saved as 'emotion_training_metrics.png'")


# # ─────────────────────────────────────────────
# # STEP 8 — Export to ONNX
# # ─────────────────────────────────────────────
# def export_onnx(weights_path, onnx_path, num_classes, device):
#     model = build_model(num_classes, device)
#     model.load_state_dict(torch.load(weights_path, map_location=device))
#     model.eval()

#     dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
#     torch.onnx.export(
#         model, dummy_input, onnx_path,
#         input_names=["input"], output_names=["output"],
#         opset_version=11, export_params=True, do_constant_folding=True,
#         training=torch.onnx.TrainingMode.EVAL,
#     )

#     if os.path.exists(onnx_path + ".data"):
#         os.remove(onnx_path + ".data")
#     print(f"ONNX model saved successfully → {onnx_path}")


# # ─────────────────────────────────────────────
# # MAIN EXECUTION BLOCK
# # ─────────────────────────────────────────────
# if __name__ == "__main__":

#     split_dataset(BASE_DIR, TRAIN_DIR, TEST_DIR)

#     train_loader, test_loader = get_dataloaders(TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE)

#     model     = build_model(NUM_CLASSES, device)
#     criterion = nn.CrossEntropyLoss()
    
#     # ANTI-OVERFITTING FIX: Added weight_decay=1e-4 for structural L2 Regularization
#     optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

#     print("\n--- Starting Training Loop ---")
#     history = train_and_evaluate(
#         model, train_loader, test_loader, criterion, optimizer, device, EPOCHS, MODEL_PATH
#     )

#     plot_metrics(history)
#     export_onnx(MODEL_PATH, ONNX_PATH, NUM_CLASSES, device)



















# # ─────────────────────────────────────────────
# # STEP 1 — Imports & Device Setup
# # ─────────────────────────────────────────────
# import os
# import shutil
# import random

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, models, transforms
# from PIL import Image
# import matplotlib.pyplot as plt

# # Check and set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# # Hardware Optimization: Enable CUDNN auto-tuner for optimal architecture performance
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = True
#     print("CUDA Optimizations Enabled (CUDNN Benchmark).")


# # ─────────────────────────────────────────────
# # STEP 2 — Paths & Hyperparameters
# # ─────────────────────────────────────────────
# BASE_DIR    = os.path.join("content", "processed_data")   
# TRAIN_DIR   = os.path.join("content", "dataset", "train")
# TEST_DIR    = os.path.join("content", "dataset", "test")
# MODEL_PATH  = "emotion_model.pth"
# ONNX_PATH   = "emotion_model.onnx"

# CLASSES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
# NUM_CLASSES = len(CLASSES)
# IMG_SIZE    = 224
# BATCH_SIZE  = 64  # Increased from 32 to 64 to leverage your 12GB VRAM on the 4070 Super
# EPOCHS      = 15  # Sufficient epochs to see convergence with metrics tracking


# # ─────────────────────────────────────────────
# # STEP 3 — Train / Test Split (80 / 20)
# # ─────────────────────────────────────────────
# def split_dataset(base_dir, train_dir, test_dir, split_ratio=0.8):
#     """Split processed_data into train/test folders if not already done."""
#     if os.path.exists(train_dir) and os.path.exists(test_dir):
#         print("Dataset already split — skipping.")
#         return

#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir,  exist_ok=True)

#     for cls in os.listdir(base_dir):
#         class_path = os.path.join(base_dir, cls)
#         if not os.path.isdir(class_path):
#             continue

#         images = os.listdir(class_path)
#         random.shuffle(images)
#         split  = int(len(images) * split_ratio)

#         for subset, imgs in [("train", images[:split]), ("test", images[split:])]:
#             dest = os.path.join("content", "dataset", subset, cls)
#             os.makedirs(dest, exist_ok=True)
#             for img in imgs:
#                 shutil.copy(os.path.join(class_path, img), os.path.join(dest, img))

#     print("Dataset split completed.")


# # ─────────────────────────────────────────────
# # STEP 4 — High-Speed DataLoaders with Augmentation
# # ─────────────────────────────────────────────
# def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, epochs, save_path):
#     history = {
#         "train_loss": [], "train_acc": [],
#         "val_loss": [], "val_acc": []
#     }

#     # Initialize at infinity so the first completed epoch is guaranteed to be lower
#     best_val_loss = float('inf') 
#     scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

#     for epoch in range(epochs):
#         # --- TRAINING ---
#         model.train()
#         running_loss, correct_train, total_train = 0.0, 0, 0

#         for images, labels in train_loader:
#             images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
#             optimizer.zero_grad(set_to_none=True)
            
#             with torch.amp.autocast(device_type=device.type):
#                 outputs = model(images)
#                 loss    = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total_train += labels.size(0)
#             correct_train += (predicted == labels).sum().item()

#         epoch_train_loss = running_loss / len(train_loader.dataset)
#         epoch_train_acc  = 100.0 * correct_train / total_train

#         # --- VALIDATION ---
#         model.eval()
#         running_val_loss, correct_val, total_val = 0.0, 0, 0

#         with torch.no_grad():
#             for images, labels in test_loader:
#                 images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
#                 with torch.amp.autocast(device_type=device.type):
#                     outputs = model(images)
#                     loss    = criterion(outputs, labels)

#                 running_val_loss += loss.item() * images.size(0)
#                 _, predicted   = torch.max(outputs, 1)
#                 total_val     += labels.size(0)
#                 correct_val   += (predicted == labels).sum().item()

#         epoch_val_loss = running_val_loss / len(test_loader.dataset)
#         epoch_val_acc  = 100.0 * correct_val / total_val

#         # Record metrics
#         history["train_loss"].append(epoch_train_loss)
#         history["train_acc"].append(epoch_train_acc)
#         history["val_loss"].append(epoch_val_loss)
#         history["val_acc"].append(epoch_val_acc)

#         # Print basic status line
#         status_msg = (f"Epoch [{epoch+1}/{epochs}] "
#                       f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || "
#                       f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
#         # CHECKPOINT LOGIC: Save weights only if validation loss gets better
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(model.state_dict(), save_path)
#             status_msg += "  --> [SAVED BEST MODEL!]"

#         print(status_msg)

#     return history


# # ─────────────────────────────────────────────
# # STEP 7 — Plotting Metrics Graph
# # ─────────────────────────────────────────────
# def plot_metrics(history):
#     epochs = range(1, len(history["train_loss"]) + 1)

#     plt.figure(figsize=(14, 5))

#     # 1. Loss Figure
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history["train_loss"], 'b-o', label='Training Loss')
#     plt.plot(epochs, history["val_loss"], 'r-o', label='Validation Loss')
#     plt.title('Loss Trends (Train vs Val)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)

#     # 2. Accuracy Figure
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history["train_acc"], 'b-o', label='Training Accuracy')
#     plt.plot(epochs, history["val_acc"], 'r-o', label='Validation Accuracy')
#     plt.title('Accuracy Trends (Train vs Val)')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig("emotion_training_metrics.png")
#     plt.show()
#     print("Metric graphs saved successfully as 'emotion_training_metrics.png'")


# # ─────────────────────────────────────────────
# # STEP 8 — Export to ONNX
# # ─────────────────────────────────────────────
# def export_onnx(model, device, onnx_path):
#     model.eval()
#     dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=["input"],
#         output_names=["output"],
#         opset_version=11,
#         export_params=True,
#         do_constant_folding=True,
#         training=torch.onnx.TrainingMode.EVAL,
#     )

#     if os.path.exists(onnx_path + ".data"):
#         os.remove(onnx_path + ".data")

#     print(f"ONNX model saved → {onnx_path}")


# # ─────────────────────────────────────────────
# # STEP 9 — Single Image Prediction Test
# # ─────────────────────────────────────────────
# def predict_image(model, img_path, classes, device):
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     image = Image.open(img_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)

#     model.eval()
#     with torch.no_grad():
#         with torch.amp.autocast(device_type=device.type):
#             output = model(image)
#         pred = torch.argmax(output, dim=1)

#     print(f"Test Inference Prediction: {classes[pred.item()]}")
#     return classes[pred.item()]


# # ─────────────────────────────────────────────
# # MAIN EXECUTION BLOCK
# # ─────────────────────────────────────────────
# if __name__ == "__main__":

#     # 1. Split dataset
#     split_dataset(BASE_DIR, TRAIN_DIR, TEST_DIR)

#     # 2. Get high-speed data loaders
#     train_loader, test_loader = get_dataloaders(
#         TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE
#     )

#     # 3. Initialize model components
#     model     = build_model(NUM_CLASSES, device)
#     criterion = nn.CrossEntropyLoss()
    
#     # 0.0005 is ideal for fine-tuning pre-trained networks safely without shattering weights
#     optimizer = optim.Adam(model.parameters(), lr=0.0005)

#     # 4. Train, track, and evaluate model
#     print("\n--- Starting Training Loop ---")
#     history = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, device, EPOCHS)

#     # 5. Plot and save metric visualization charts
#     plot_metrics(history)

#     # 6. Save native PyTorch model weights
#     torch.save(model.state_dict(), MODEL_PATH)
#     print(f"Model saved → {MODEL_PATH}")

#     # 7. Export runtime optimized graph via ONNX
#     export_onnx(model, device, ONNX_PATH)

#     # 8. Single image structural verification pass
#     test_happy = os.path.join(TEST_DIR, "happy")
#     if os.path.exists(test_happy) and len(os.listdir(test_happy)) > 0:
#         sample_img = os.path.join(test_happy, os.listdir(test_happy)[0])
#         predict_image(model, sample_img, CLASSES, device)