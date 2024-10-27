import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models
import segmentation_models_pytorch as smp

# Clear GPU cache
torch.cuda.empty_cache()

def convrelu(in_channels, out_channels, kernel, padding):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
    nn.ReLU(inplace=True),
  )


class ResNetUNet(nn.Module):
  def __init__(self, n_class):
    super().__init__()

    self.base_model = torchvision.models.resnet34(pretrained=True)
    self.base_layers = list(self.base_model.children())

    self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
    self.layer0_1x1 = convrelu(64, 64, 1, 0)
    self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
    self.layer1_1x1 = convrelu(64, 64, 1, 0)
    self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
    self.layer2_1x1 = convrelu(128, 128, 1, 0)
    self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
    self.layer3_1x1 = convrelu(256, 256, 1, 0)
    self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
    self.layer4_1x1 = convrelu(512, 512, 1, 0)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
    self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
    self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
    self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

    self.conv_original_size0 = convrelu(3, 64, 3, 1)
    self.conv_original_size1 = convrelu(64, 64, 3, 1)
    self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

    self.conv_last = nn.Conv2d(64, n_class, 1)

  def forward(self, input):
    x_original = self.conv_original_size0(input)
    x_original = self.conv_original_size1(x_original)

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)
    layer3 = self.layer3(layer2)
    layer4 = self.layer4(layer3)

    layer4 = self.layer4_1x1(layer4)
    x = self.upsample(layer4)
    layer3 = self.layer3_1x1(layer3)

    if x.size()[2:] != layer3.size()[2:]:
        x = nn.functional.interpolate(x, size=layer3.size()[2:], mode='bilinear', align_corners=True)

    x = torch.cat([x, layer3], dim=1)
    x = self.conv_up3(x)

    x = self.upsample(x)
    layer2 = self.layer2_1x1(layer2)

    if x.size()[2:] != layer2.size()[2:]:
        x = nn.functional.interpolate(x, size=layer2.size()[2:], mode='bilinear', align_corners=True)

    x = torch.cat([x, layer2], dim=1)
    x = self.conv_up2(x)

    x = self.upsample(x)
    layer1 = self.layer1_1x1(layer1)

    if x.size()[2:] != layer1.size()[2:]:
        x = nn.functional.interpolate(x, size=layer1.size()[2:], mode='bilinear', align_corners=True)

    x = torch.cat([x, layer1], dim=1)
    x = self.conv_up1(x)

    x = self.upsample(x)
    layer0 = self.layer0_1x1(layer0)

    if x.size()[2:] != layer0.size()[2:]:
        x = nn.functional.interpolate(x, size=layer0.size()[2:], mode='bilinear', align_corners=True)

    x = torch.cat([x, layer0], dim=1)
    x = self.conv_up0(x)

    x = self.upsample(x)

    if x.size()[2:] != x_original.size()[2:]:
        x = nn.functional.interpolate(x, size=x_original.size()[2:], mode='bilinear', align_corners=True)

    x = torch.cat([x, x_original], dim=1)
    x = self.conv_original_size2(x)

    out = self.conv_last(x)

    return out


import os
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, jaccard_score

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.masks_files = sorted(os.listdir(os.path.join(root, "masks")))

    def __len__(self):
        return len(self.imgs_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, "images", self.imgs_files[index])
        mask_path = os.path.join(self.root, "masks", self.masks_files[index])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        return image, mask

#    def collate_fn(self, batch):
#        return tuple(zip(*batch))

gpu_index = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using GPU {gpu_index}:", torch.cuda.get_device_name(device))

transform = transforms.Compose([
    transforms.ToTensor()
])

BONELINE_FOLDER_TRAIN = 'Data_boneline/train/'
BONELINE_FOLDER_VAL = 'Data_boneline/val/'
BONELINE_FOLDER_TEST = 'Data_boneline/test/'

dataset_train = CustomDataset(BONELINE_FOLDER_TRAIN, transform=False)
dataset_val = CustomDataset(BONELINE_FOLDER_VAL, transform=False)
dataset_test = CustomDataset(BONELINE_FOLDER_TEST, transform=False)

data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# **Model Initialize**

model = ResNetUNet(n_class=1)

#model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000007, weight_decay=1e-6)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

# **Training**

model.train()

n_epochs = 500
final_epoch = 0

train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 10
best_model_state = None
early_stopped = False

for epoch in range(n_epochs):

    model.train()
    total_train_loss = 0.0
    for images, masks in data_loader_train:

        optimizer.zero_grad()

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    lr_scheduler.step()

    avg_train_loss = total_train_loss / len(data_loader_train)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss}")

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in data_loader_val:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks)

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(data_loader_val)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss}")

    lr_scheduler.step()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        print(f"Now epoch number is {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve.")
            early_stopped = True
            break

if early_stopped and best_model_state is not None:
    model.load_state_dict(best_model_state)

save_path = 'unet_resnet_encoder.pth'
torch.save(model, save_path)


# **Evaluation**
# Metric functions
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.sigmoid(output)  # Ensure output is in range [0, 1]
        output = (output > 0.5).float()  # Threshold to binary (0 or 1)
        correct = (output == mask).sum().item()
        total = mask.numel()
        return correct / total

def dice_coefficient(output, mask):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        mask = mask.float()
        intersection = (output * mask).sum().item()
        return (2.0 * intersection) / (output.sum().item() + mask.sum().item() + 1e-8)

def precision(output, mask):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        mask = mask.float()
        true_positive = (output * mask).sum().item()
        false_positive = (output * (1 - mask)).sum().item()
        return true_positive / (true_positive + false_positive + 1e-8)

def recall(output, mask):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        mask = mask.float()
        true_positive = (output * mask).sum().item()
        false_negative = ((1 - output) * mask).sum().item()
        return true_positive / (true_positive + false_negative + 1e-8)

def iou(output, mask):
    with torch.no_grad():
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        mask = mask.float()
        intersection = (output * mask).sum().item()
        union = output.sum().item() + mask.sum().item() - intersection
        return intersection / (union + 1e-8)

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_pixel_accuracy = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    total_batches = len(dataloader)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            total_pixel_accuracy += pixel_accuracy(outputs, masks)
            total_dice += dice_coefficient(outputs, masks)
            total_precision += precision(outputs, masks)
            total_recall += recall(outputs, masks)
            total_iou += iou(outputs, masks)

    avg_pixel_accuracy = total_pixel_accuracy / total_batches
    avg_dice = total_dice / total_batches
    avg_precision = total_precision / total_batches
    avg_recall = total_recall / total_batches
    avg_iou = total_iou / total_batches

    return avg_pixel_accuracy, avg_dice, avg_precision, avg_recall, avg_iou

print("TRAIN")
avg_pixel_accuracy, avg_dice, avg_precision, avg_recall, avg_iou = evaluate_model(model, data_loader_train, device)
print(f"Pixel Accuracy: {avg_pixel_accuracy:.4f}, Dice Coefficient: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}")

print("VALIDATION")
avg_pixel_accuracy, avg_dice, avg_precision, avg_recall, avg_iou = evaluate_model(model, data_loader_val, device)
print(f"Pixel Accuracy: {avg_pixel_accuracy:.4f}, Dice Coefficient: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}")

print("TEST")
avg_pixel_accuracy, avg_dice, avg_precision, avg_recall, avg_iou = evaluate_model(model, data_loader_test, device)
print(f"Pixel Accuracy: {avg_pixel_accuracy:.4f}, Dice Coefficient: {avg_dice:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}")
