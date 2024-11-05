import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from networks import AlexNet  

batch_size_train = 16
batch_size_test = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20

# Define dataset class for synthetic data
class SyntheticDataset(Dataset):
    def __init__(self, root_dir, num_classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for class_idx in range(num_classes):
            class_dir = os.path.join(root_dir, str(class_idx))
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define real MHIST dataset loader with label mapping
class RealMHISTDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

        # Map categorical labels to integers
        label_mapping = {'SSA': 0, 'HP': 1}
        self.images = []
        self.labels = []

        # Populate image paths and labels with the mapped values
        for idx, row in self.annotations.iterrows():
            img_name = row['Image Name']
            label_str = row['Majority Vote Label']
            
            # Check if label is in our mapping dictionary
            if label_str in label_mapping:
                int_label = label_mapping[label_str]
                self.images.append(img_name)
                self.labels.append(int_label)
            else:
                print(f"Warning: Unrecognized label '{label_str}' in row {idx}, skipping.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Data transformations
transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3 channels for AlexNet
    transforms.Resize((32, 32)),  # Resize to 32x32 for AlexNet
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

transform_mhist = transforms.Compose([
    transforms.Resize((64, 64)),  # Match the expected input size for MHIST
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load synthetic and real datasets
mnist_synthetic_train = SyntheticDataset(root_dir='./mnist_synthesis', num_classes=10, transform=transform_mnist)
mhist_synthetic_train = SyntheticDataset(root_dir='./mhist_synthesis', num_classes=2, transform=transform_mhist)

real_mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
real_mhist_test = RealMHISTDataset(img_dir='./mhist_dataset/mhist_images', csv_file='./mhist_dataset/annotations.csv', transform=transform_mhist)

# Data loaders
mnist_train_loader = DataLoader(mnist_synthetic_train, batch_size=batch_size_train, shuffle=True)
mnist_test_loader = DataLoader(real_mnist_test, batch_size=batch_size_test, shuffle=False)

mhist_train_loader = DataLoader(mhist_synthetic_train, batch_size=batch_size_train, shuffle=True)
mhist_test_loader = DataLoader(real_mhist_test, batch_size=batch_size_test, shuffle=False)

# Modify AlexNet to dynamically handle different input sizes
class AlexNetCustom(AlexNet):
    def __init__(self, channel, num_classes, input_size=(32, 32)):
        super(AlexNetCustom, self).__init__(channel, num_classes)
        # Calculate flattened feature size after convolutional layers
        dummy_input = torch.zeros(1, channel, *input_size)
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.view(-1).size(0)
        # Replace the fully connected layer with dynamically calculated input size
        self.fc = nn.Linear(flattened_size, num_classes)

# Function to train and evaluate model, with timing
def train_and_evaluate(model, train_loader, test_loader, epochs=20, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Track training time
    start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy, training_time

# Select model and train on synthetic MNIST, evaluate on real MNIST
print("Training on synthetic MNIST, evaluating on real MNIST:")
model_mnist = AlexNetCustom(channel=3, num_classes=10, input_size=(32, 32))  # Using 3 channels for MNIST to match RGB format
mnist_accuracy, mnist_training_time = train_and_evaluate(model_mnist, mnist_train_loader, mnist_test_loader, epochs=epochs)
print(f"MNIST - Training Time: {mnist_training_time:.2f} seconds, Test Accuracy: {mnist_accuracy:.2f}%")

# Select model and train on synthetic MHIST, evaluate on real MHIST
print("\nTraining on synthetic MHIST, evaluating on real MHIST:")
model_mhist = AlexNetCustom(channel=3, num_classes=2, input_size=(64, 64))  # MHIST is already RGB and larger input size
mhist_accuracy, mhist_training_time = train_and_evaluate(model_mhist, mhist_train_loader, mhist_test_loader, epochs=epochs)
print(f"MHIST - Training Time: {mhist_training_time:.2f} seconds, Test Accuracy: {mhist_accuracy:.2f}%")