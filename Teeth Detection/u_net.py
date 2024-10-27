import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

torch.cuda.empty_cache()

### -------- U - Net

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

### -------- U - Net with Attention 

# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class encoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.conv = conv_block(in_c, out_c)
#         self.pool = nn.MaxPool2d((2, 2))

#     def forward(self, x):
#         s = self.conv(x)
#         p = self.pool(s)
#         return s, p

# class attention_gate(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.Wg = nn.Sequential(
#             nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_c)
#         )
#         self.Ws = nn.Sequential(
#             nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_c)
#         )
#         self.relu = nn.ReLU(inplace=True)
#         self.output = nn.Sequential(
#             nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )

#     def forward(self, g, s):
#         Wg = self.Wg(g)
#         Ws = self.Ws(s)

#         # Padding to match the size
#         diffY = Wg.size()[2] - Ws.size()[2]
#         diffX = Wg.size()[3] - Ws.size()[3]

#         Ws = nn.functional.pad(Ws, [diffX // 2, diffX - diffX // 2,
#                                     diffY // 2, diffY - diffY // 2])

#         out = self.relu(Wg + Ws)
#         out = self.output(out)
#         return out * s

# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.ag = attention_gate(in_c, out_c)
#         self.c1 = conv_block(in_c[0]+out_c, out_c)

#     def forward(self, x, s):
#         x = self.up(x)

#         # Padding to match the size of x and s
#         diffY = s.size()[2] - x.size()[2]
#         diffX = s.size()[3] - x.size()[3]

#         x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
#                                   diffY // 2, diffY - diffY // 2])

#         s = self.ag(x, s)
#         x = torch.cat([x, s], axis=1)
#         x = self.c1(x)
#         return x

# class attention_unet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.e1 = encoder_block(3, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)

#         self.b1 = conv_block(256, 512)

#         self.d1 = decoder_block([512, 256], 256)
#         self.d2 = decoder_block([256, 128], 128)
#         self.d3 = decoder_block([128, 64], 64)

#         self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

#     def forward(self, x):
#         s1, p1 = self.e1(x)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)

#         b1 = self.b1(p3)

#         d1 = self.d1(b1, s3)
#         d2 = self.d2(d1, s2)
#         d3 = self.d3(d2, s1)

#         output = self.output(d3)
#         return output

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.ag = attention_gate(in_c, out_c)
        self.c1 = conv_block(in_c[0]+out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        self.b1 = conv_block(256, 512)

        self.d1 = decoder_block([512, 256], 256)
        self.d2 = decoder_block([256, 128], 128)
        self.d3 = decoder_block([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output


import os
import torch
import torch.nn as nn
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
        else:
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
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# # For the Teeth Maks
BONELINE_FOLDER_TRAIN = 'Data/train/'
BONELINE_FOLDER_VAL = 'Data/val/'
BONELINE_FOLDER_TEST = 'Data/test/'

dataset_train = CustomDataset(BONELINE_FOLDER_TRAIN, transform=transform)
dataset_val = CustomDataset(BONELINE_FOLDER_VAL, transform=transform)
dataset_test = CustomDataset(BONELINE_FOLDER_TEST, transform=transform)

data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# **Model Initialize**

in_channels = 3
out_channels = 1
model = UNET(in_channels, out_channels)


#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000001, weight_decay=1e-6)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)

# **Training**

model.train()

n_epochs = 1000
final_epoch = 0

train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 20
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

    #lr_scheduler.step()

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

#    lr_scheduler.step()

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

save_path = 'unet_model.pth'
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
