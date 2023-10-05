import torch
import torchvision.models.segmentation
from torchvision import transforms
from dataset_cl import CustomSegmentationDataset
from DiceLoss import DiceLoss
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import cv2
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Объявим аугментации
data_transforms = {
    'image': transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
}

dataset_root = 'train_data'
dataset = CustomSegmentationDataset(root=dataset_root, transform=data_transforms)

#Разобьем данные на тренировочную и тестовую выборки
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


#Загрузим предобученную Unet с довольно простым бэкбоном Resnet50
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
#Немного подправим модель под нашу задачу семантической сегментации с одним классом
num_classes = 1
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
model.aux_classifier = None
model.to(device)

# Зададим параметры обучения и запустим цикл обучения
batch_size = 12
learning_rate = 0.001
num_epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = DiceLoss() #так же были протестированы BCELoss, JaccardLoss

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        # Forward pass
        images, masks = batch['image'].to(device), batch['mask'].to(device)

        outputs = model(images)['out']

        loss = criterion(outputs, masks)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    best_val_loss = float('inf')
    # Initialize metrics for evaluation (IoU, Dice, etc.)

    with torch.no_grad():
        for batch in val_loader:
            # Forward pass and evaluation metrics computation
            images, masks = batch['image'].to(device), batch['mask'].to(device)

            # Forward pass
            outputs = model(images)['out']

            # Calculate the loss
            loss = criterion(outputs, masks)

            if loss < best_val_loss: #будем сохранять лучшие веса модели
                best_val_loss = loss
                torch.save(model.state_dict(), 'best_model2.pth')

            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader)}")

#Попробуем оценить полученную модель

model.eval()
test_transforms = {
    'image': transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'mask': transforms.Compose([
        transforms.Resize((640, 360)),
        transforms.ToTensor()
    ])
}

test_dataset = CustomSegmentationDataset(root='test_data', transform=test_transforms)

validation_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
output_folder = 'preds'
# Initialize variables to calculate metrics
iou_total = 0.0
dice_total = 0.0
pixel_accuracy_total = 0.0


for batch in validation_loader:
    with torch.no_grad():

        image, target = batch['image'].to(device), batch['mask'].to(device)

        
        output = model(image)['out']

        
        pred_labels = output.argmax(1)
        pred_probs = torch.softmax(output, dim=1)[0]
        pl = pred_labels.cpu().numpy()
        predicted_class = np.uint8(pl[0])
        # Calculate intersection over union (IoU)
        intersection = torch.logical_and(pred_labels, target).sum().item()
        union = torch.logical_or(pred_labels, target).sum().item()
        iou = intersection / union
        iou_total += iou

        
        dice = (2 * intersection) / (pred_labels.sum().item() + target.sum().item())
        dice_total += dice

        
        pixel_accuracy = (pred_labels == target).sum().item() / target.numel()
        pixel_accuracy_total += pixel_accuracy

        mask_path = os.path.join(output_folder, f'_mask.png')
        cv2.imwrite(mask_path, predicted_class)


num_samples = 10
average_iou = iou_total / num_samples
average_dice = dice_total / num_samples
average_pixel_accuracy = pixel_accuracy_total / num_samples

print(f"Average IoU: {average_iou:.4f}")
print(f"Average Dice: {average_dice:.4f}")
print(f"Average Pixel Accuracy: {average_pixel_accuracy:.4f}")
