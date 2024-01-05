import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ----------------------------------------------
# 0. Define a custom dataset
# ----------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)['dataset']['samples']
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        resize = transforms.Resize((256, 256))
        # center_crop = transforms.CenterCrop(224)
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['attributes']['image']['url'].split('/')[-1])
        img = Image.open(img_path).convert('RGB')

        # Get the labels
        bitmap_url = item['labels']['ground-truth']['attributes']['segmentation_bitmap']['url']

        # Load the bitmap masks
        bitmap_path = os.path.join(self.img_dir, bitmap_url.split('/')[-1])
        bitmap = Image.open(bitmap_path).convert('L')
        bitmap = resize(bitmap)
        # bitmap = center_crop(bitmap)

        # Convert the bitmap to a into a 2D matrix
        mask = np.array(bitmap)
        mask = torch.from_numpy(mask).long()

        img = resize(img)
        # img = center_crop(img)

        if self.transform:
            img = self.transform(img)

        return img, mask

# ----------------------------------------------
# 1. Load and pre-process the images, and create the dataset
# ----------------------------------------------

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256 pixels
    transforms.ToTensor(),  # Convert to PyTorch Tensor data type
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and standard deviation
])

# Create the dataset
dataset = CustomDataset('gummi_bears-v0.1.json', './images', transform=transform)

# Create the data loader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print('Number of samples: ', len(dataset))

# ----------------------------------------------
# 2. Define a U-Net model
# ----------------------------------------------

import segmentation_models_pytorch as smp

# Create U-Net model
model = smp.Unet(
   encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
   encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
   in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
   classes=5,                      # model output channels (number of classes in your dataset)
)

# ----------------------------------------------
# 3. Train the model
# ----------------------------------------------

# Define a loss function
criterion = torch.nn.CrossEntropyLoss()

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define number of epochs
num_epochs = 10

# Assume that we are on a CUDA machine, then this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the GPU
model = model.to(device)

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')