import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

torch.cuda.empty_cache()

### -------- multiresunet

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, act=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c)
        ]
        if act == True:
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class multires_block(nn.Module):
    def __init__(self, in_c, out_c, alpha=1.67):
        super().__init__()

        W = out_c * alpha
        self.c1 = conv_block(in_c, int(W*0.167))
        self.c2 = conv_block(int(W*0.167), int(W*0.333))
        self.c3 = conv_block(int(W*0.333), int(W*0.5))

        nf = int(W*0.167) + int(W*0.333) + int(W*0.5)
        self.b1 = nn.BatchNorm2d(nf)
        self.c4 = conv_block(in_c, nf)
        self.relu = nn.ReLU(inplace=True)
        self.b2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        x0 = x
        x1 = self.c1(x0)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        xc = torch.cat([x1, x2, x3], dim=1)
        xc = self.b1(xc)

        sc = self.c4(x0)
        x = self.relu(xc + sc)
        x = self.b2(x)
        return x

class res_path_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = conv_block(in_c, out_c, act=False)
        self.s1 = conv_block(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x1 = self.c1(x)
        s1 = self.s1(x)
        x = self.relu(x1 + s1)
        x = self.bn(x)
        return x

class res_path(nn.Module):
    def __init__(self, in_c, out_c, length):
        super().__init__()

        layers = []
        for i in range(length):
            layers.append(res_path_block(in_c, out_c))
            in_c = out_c

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

def cal_nf(ch, alpha=1.67):
    W = ch * alpha
    return int(W*0.167) + int(W*0.333) + int(W*0.5)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, length):
        super().__init__()

        self.c1 = multires_block(in_c, out_c)
        nf = cal_nf(out_c)
        self.s1 = res_path(nf, out_c, length)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.c1(x)
        s = self.s1(x)
        p = self.pool(x)
        return s, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.ConvTranspose2d(in_c[0], out_c, kernel_size=2, stride=2, padding=0)
        self.c2 = multires_block(out_c+in_c[1], out_c)

    def forward(self, x, s):
        x = self.c1(x)
        x = torch.cat([x, s], dim=1)
        x = self.c2(x)
        return x

class build_multiresunet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 32, 4)
        self.e2 = encoder_block(cal_nf(32), 64, 3)
        self.e3 = encoder_block(cal_nf(64), 128, 2)
        self.e4 = encoder_block(cal_nf(128), 256, 1)

        """ Bridge """
        self.b1 = multires_block(cal_nf(256), 512)

        """ Decoder """
        self.d1 = decoder_block([cal_nf(512), 256], 256)
        self.d2 = decoder_block([cal_nf(256), 128], 128)
        self.d3 = decoder_block([cal_nf(128), 64], 64)
        self.d4 = decoder_block([cal_nf(64), 32], 32)

        """ Output """
        self.output = nn.Conv2d(cal_nf(32), 1, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        b1 = self.b1(p4)

        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        output = self.output(d4)
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
model = build_multiresunet()

#model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.000007, weight_decay=1e-6)
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
