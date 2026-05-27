# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

The problem is to design and develop a Convolutional Deep Neural Network (CNN) that can automatically classify grayscale images into predefined categories. The model must learn important spatial features such as edges, textures, and shapes from image data and accurately predict the correct class label.


## Neural Network Model
<img width="1088" height="768" alt="image" src="https://github.com/user-attachments/assets/3313bb29-72b5-44e9-9cb9-a1f102a0225b" />


## DESIGN STEPS
### STEP 1: 

Load and Preprocess Data

### STEP 2: 

Get the shape of the first image in the training dataset


### STEP 3: 

Get the shape of the first image in the test dataset

### STEP 4: 

Train the model

### STEP 5: 

test the model

### STEP 6: 

predict on a single image and display the image



## PROGRAM

### Name: SURUTHI S

### Register Number:212224220114

``` python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data

# Define transformations for images
transform = transforms.Compose([
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=transform, download=True)

# Get the shape of the first image in the training dataset
image, label = train_dataset[0]
print(image.shape)
print(len(train_dataset))

# Get the shape of the first image in the test dataset
image, label = test_dataset[0]
print(image.shape)
print(len(test_dataset))

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        # conv2 should take 32 channels as input from conv1's output after pooling
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        # conv3 should take 64 channels as input from conv2's output after pooling
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # Calculate input size for fc1: 128 channels * 3x3 feature map after 3 pooling layers
        # (28 -> 14 -> 7 -> 3) based on kernel_size=2, stride=2
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: SURUTHI S')
print('Register Number: 212224220114')
summary(model, input_size=(1, 28, 28))

# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Name: SURUTHI S')
    print('Register Number: 212224220114')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Train the model
train_model(model, train_loader)

## Step 4: Test the Model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print('Name: SURUTHI S')
    print('Register Number: 212224220114')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    print('Name: SURUTHI S')
    print('Register Number: 212224220114')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print('Name: SURUTHI S')
    print('Register Number: 212224220114')
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Evaluate the model
test_model(model, test_loader)
## Step 5: Predict on a Single Image
import matplotlib.pyplot as plt
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        _, predicted = torch.max(output, 1)
    class_names = dataset.classes

    # Display the image
    print('Name: SURUTHI S')
    print('Register Number:212224220114')
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')

# Example Prediction
predict_image(model, image_index=331, dataset=test_dataset)



```

## OUTPUT
### Training Loss per Epoch

<img width="438" height="289" alt="image" src="https://github.com/user-attachments/assets/40301973-43af-402e-aea6-850642b3c459" />


## Confusion Matrix


<img width="962" height="802" alt="image" src="https://github.com/user-attachments/assets/deea82df-24ad-4b6c-afcd-57dc28b92fb4" />


## Classification Report
<img width="573" height="399" alt="image" src="https://github.com/user-attachments/assets/3b93ca14-7baa-4bc0-9160-da635df34d42" />


### New Sample Data Prediction

<img width="572" height="614" alt="image" src="https://github.com/user-attachments/assets/08304e28-5789-407a-960f-04a40ce10434" />


## RESULT
The convolutional neural network (CNN) model was successfully developed and trained for image classification
