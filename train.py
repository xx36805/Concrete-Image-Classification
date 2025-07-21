import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import os

from MMC.mmc import mmc, prompt_load
from tools import aliyun_oss, qwen_vl_max

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Data Loading
FILE_PATH = os.path.join(os.getcwd(), 'D:\work_software\PyCharmWork\HNTCV\Concrete-Image-Classification-main\our_dataset')

# 2. Data Preparation
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
SEED = 12345
# Training Loop
EPOCHS = 10

# Define data transformations
train_transforms = transforms.Compose([
    # inception_v3
    # transforms.Resize((299, 299)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    # transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = datasets.ImageFolder(os.path.join(FILE_PATH, 'condef/train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(FILE_PATH, 'condef/test'), transform=val_test_transforms)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Define model
model_name = 'mobilenet_v2'
model = models.mobilenet_v2(pretrained=True)
if model_name == "inception_v3":
    model.aux_logits = False 

# Freeze base model layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
if model_name == "VGG19":
    # VGG19
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(train_dataset.classes))
elif model_name in ["resnet50", "inception_v3"]:
    # resnet50 inception_v3
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
else:
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
if model_name in ["resnet50", "inception_v3"]:
    # resnet50 inception_v3
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
else:
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)


# TensorBoard Setup
log_dir = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    temp_epoch_acc = 0
    val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in tqdm(train_loader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.5f}, Accuracy: {epoch_acc:.5f}")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        total_visual_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                for value in predicted:
                    total_visual_preds.append(value.item())
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct_preds / total_preds
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print("---------------------Output the prediction results of visual modality-------------------")
        print(total_visual_preds)
        if temp_epoch_acc < val_acc:
            print("Save the best model...")
            temp_epoch_acc = val_acc
            # Save the model
            save_path = os.path.join(os.getcwd(), 'saved_model', 'saved_models.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")



def text_classifier(train_loader, epochs, all_prompts, all_label):
    for epoch in range(epochs):
        total_text_preds = []
        for images, labels in train_loader:
            
            # batch_indices = train_loader.dataset.samples

            #
            image_paths = [train_loader.dataset.samples[idx][0] for idx in labels.tolist()]

            # upload aliyun oss
            for i in range(len(image_paths)):
                label_str = str(labels[i].item())
                print(f"Path: {image_paths[i]}, Label: {label_str}")
                filename = label_str + "_" + image_paths[i][image_paths[i].rfind("\\") + 1:]
                aliyun_oss.oss_upload(image_paths[i], filename)
                oss_path = "https://hntcv.oss-cn-beijing.aliyuncs.com/HNT_IMG/" + filename

                img_description = qwen_vl_max.call_with_messages(oss_path, 0.1, 50, 0.1)

                pred, correct = mmc(img_description, label_str, all_prompts, all_label)
                total_text_preds.append(pred)
        print("---------------------Output the predicted results of text modality-------------------")
        print(total_text_preds)


if __name__ == '__main__':
    # all_prompts, all_label = prompt_load()
    # text_classifier(train_loader, EPOCHS, all_prompts, all_label)
    train_model(model, train_loader, val_loader, loss_fn, optimizer, EPOCHS)
