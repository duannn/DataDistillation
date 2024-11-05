import os
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import time

batch_size_train = 16
batch_size_test = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 50

# Define SyntheticDataset class for loading synthetic datasets
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

# Define transformations for MNIST and MHIST datasets
transform_mnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert MNIST to 3 channels for compatibility
    transforms.Resize((32, 32)),  # Resize for consistency in CNN models
    transforms.ToTensor(),
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

transform_mhist = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize for compatibility with the search space
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Typical normalization for RGB
])

# Load the synthetic MNIST and MHIST datasets for training
mnist_synthetic_train = SyntheticDataset(root_dir='./mnist_synthesis', num_classes=10, transform=transform_mnist)
mhist_synthetic_train = SyntheticDataset(root_dir='./mhist_synthesis', num_classes=2, transform=transform_mhist)

# Loaders for training on synthetic data
mnist_train_loader = DataLoader(mnist_synthetic_train, batch_size=batch_size_train, shuffle=True)
mhist_train_loader = DataLoader(mhist_synthetic_train, batch_size=batch_size_train, shuffle=True)

# Load real MNIST test dataset
mnist_real_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_test_loader = DataLoader(mnist_real_test, batch_size=batch_size_test, shuffle=False)

# Define RealMHISTDataset class for loading real MHIST dataset
class RealMHISTDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        label_mapping = {'SSA': 0, 'HP': 1}  # Map labels to integers
        self.images = []
        self.labels = []

        for idx, row in self.annotations.iterrows():
            img_name = row['Image Name']
            label_str = row['Majority Vote Label']
            if label_str in label_mapping:
                int_label = label_mapping[label_str]
                self.images.append(os.path.join(self.img_dir, img_name))
                self.labels.append(int_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load real MHIST test dataset
mhist_real_test = RealMHISTDataset(
    img_dir='./mhist_dataset/mhist_images',
    csv_file='./mhist_dataset/annotations.csv',
    transform=transform_mhist
)
mhist_test_loader = DataLoader(mhist_real_test, batch_size=batch_size_test, shuffle=False)

# Define a simple search space with combinations of layers and parameters
search_space = [
    {'conv_layers': 1, 'conv_channels': [32], 'fc_layers': 1, 'fc_units': [128]},
    {'conv_layers': 2, 'conv_channels': [32, 64], 'fc_layers': 1, 'fc_units': [128]},
    {'conv_layers': 2, 'conv_channels': [64, 128], 'fc_layers': 2, 'fc_units': [128, 64]},
    {'conv_layers': 3, 'conv_channels': [32, 64, 128], 'fc_layers': 1, 'fc_units': [256]},
]

# Define a function to build a CNN model based on the search space configuration
class CNNModel(nn.Module):
    def __init__(self, config, num_classes):
        super(CNNModel, self).__init__()
        layers = []
        in_channels = 3  # RGB channels for compatibility
        for i in range(config['conv_layers']):
            layers.append(nn.Conv2d(in_channels, config['conv_channels'][i], kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = config['conv_channels'][i]
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate the flattened feature size after convolution layers
        dummy_input = torch.zeros(1, 3, 32, 32)
        conv_output_size = self.conv(dummy_input).view(-1).size(0)
        
        # Fully connected layers
        fc_layers = []
        in_features = conv_output_size
        for j in range(config['fc_layers']):
            fc_layers.append(nn.Linear(in_features, config['fc_units'][j]))
            fc_layers.append(nn.ReLU())
            in_features = config['fc_units'][j]
        
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Function to evaluate a given configuration
def evaluate_configuration(config, train_loader, test_loader, num_classes, epochs=10, lr=0.001):
    model = CNNModel(config, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Perform random search over the search space and select the configuration with the highest average accuracy
def random_search_with_highest_avg_accuracy(train_loader, test_loader, num_classes, search_space, trials=10):
    accuracy_results = {i: [] for i in range(len(search_space))}  # Track accuracies for each config

    # Perform NAS random search with multiple trials
    for trial in range(trials):
        for idx, config in enumerate(search_space):
            accuracy = evaluate_configuration(config, train_loader, test_loader, num_classes)
            accuracy_results[idx].append(accuracy)
            #print(f"Trial {trial+1} | Config {idx+1}: {config} | Accuracy: {accuracy:.2f}%")

    # Calculate the average accuracy for each configuration
    avg_accuracies = {idx: sum(acc_list) / len(acc_list) for idx, acc_list in accuracy_results.items()}
    
    # Select the configuration with the highest average accuracy
    best_config_idx = max(avg_accuracies, key=avg_accuracies.get)
    best_config = search_space[best_config_idx]
    best_avg_accuracy = avg_accuracies[best_config_idx]

    print(f"\nSelected Best Configuration from NAS: Config{best_config_idx + 1}: {best_config} with Highest Average Accuracy: {best_avg_accuracy:.2f}%")

    # Verification Step: Perform 5 trials for each configuration to confirm best choice
    verification_results = {}
    for idx, config in enumerate(search_space):
        trial_accuracies = []
        for _ in range(5):  # 5 trials per configuration
            accuracy = evaluate_configuration(config, train_loader, test_loader, num_classes)
            trial_accuracies.append(accuracy)
        avg_verification_accuracy = sum(trial_accuracies) / 5
        verification_results[f"Config {idx + 1}"] = avg_verification_accuracy
        print(f"Config {idx + 1} Verification Average Accuracy: {avg_verification_accuracy:.2f}%")

    best_verified_config = max(verification_results, key=verification_results.get)
    best_verified_accuracy = verification_results[best_verified_config]
    
    print(f"\nBest Verified Configuration: {best_verified_config} with Verification Accuracy: {best_verified_accuracy:.2f}%")
    return best_config, best_avg_accuracy, verification_results

print("Searching best architecture for MNIST synthetic dataset with highest average accuracy and verification:")
best_mnist_config, best_mnist_avg_accuracy, mnist_verification_results = random_search_with_highest_avg_accuracy(
    mnist_train_loader, mnist_test_loader, num_classes=10, search_space=search_space, trials=5
)

print("\nSearching best architecture for MHIST synthetic dataset with highest average accuracy and verification:")
best_mhist_config, best_mhist_avg_accuracy, mhist_verification_results = random_search_with_highest_avg_accuracy(
    mhist_train_loader, mhist_test_loader, num_classes=2, search_space=search_space, trials=5
)





