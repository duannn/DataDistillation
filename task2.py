import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
from networks import ConvNet  

batch_size_train = 16
batch_size_test = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20
num_classes = 10
images_per_class = 10  

# Transformation for MNIST images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Generate condensed images using PAD with real image initialization
def generate_condensed_images_PAD(root_dir='./PAD_synthesis', visualize=True):
    os.makedirs(root_dir, exist_ok=True)
    for class_idx in range(num_classes):
        class_dir = os.path.join(root_dir, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)

    real_mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    real_mnist_loader = DataLoader(real_mnist_train, batch_size=1, shuffle=True)
    initial_images = {i: [] for i in range(num_classes)}

    for image, label in real_mnist_loader:
        label = label.item()
        if len(initial_images[label]) < images_per_class:
            initial_images[label].append(image.squeeze(0).to(device))
        if all(len(v) == images_per_class for v in initial_images.values()):
            break  

    # Initialize synthetic images 
    synthetic_images = []
    synthetic_labels = []
    optimizable_images = []

    for class_idx in range(num_classes):
        class_images = []
        for i in range(images_per_class):
            image = initial_images[class_idx][i].clone().requires_grad_(True)  
            class_images.append(image)
            optimizable_images.append(image)  
        
        synthetic_images.append(class_images)
        synthetic_labels.append(torch.full((images_per_class,), class_idx, device=device, dtype=torch.long))

    # Define ConvNet3 model and criterion for PAD optimization
    model = ConvNet(channel=1, num_classes=num_classes, net_width=128, net_depth=3,
                    net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)).to(device)
    model.eval()  # The model is used in eval mode during synthesis
    criterion = nn.CrossEntropyLoss()

    # Optimizer for synthetic data
    optimizer = optim.SGD(optimizable_images, lr=0.01)  # Optimizing each individual image tensor

    # Perform optimization steps to refine the synthetic images
    opt_step = 100
    for step in range(opt_step):  
        optimizer.zero_grad()
        loss = 0
        for class_idx in range(num_classes):
            class_images = torch.stack([img for img in synthetic_images[class_idx]]).to(device)
            labels = synthetic_labels[class_idx]
            outputs = model(class_images)
            loss += criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # if step % 10 == 0:
        #     print(f"Optimization Step [{step}/{opt_step}], Loss: {loss.item():.4f}")

    # Save generated images to disk and visualize
    for class_idx in range(num_classes):
        class_images = synthetic_images[class_idx]
        for idx, img in enumerate(class_images):
            img_path = os.path.join(root_dir, str(class_idx), f'synthesized_{idx}.png')
            save_image(img.detach().cpu(), img_path)

    print(f"Generated and saved condensed images to {root_dir}")

    # Visualization of condensed images
    if visualize:
        filename="PAD_MNIST_Synthesis.png"
        fig, axs = plt.subplots(num_classes, images_per_class, figsize=(images_per_class, num_classes))
        fig.suptitle("MNIST Condensed Images using PAD", fontsize=16)
        for class_idx in range(num_classes):
            for img_idx, img in enumerate(synthetic_images[class_idx]):
                axs[class_idx, img_idx].imshow(img.detach().cpu().squeeze(), cmap="gray")
                axs[class_idx, img_idx].axis("off")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

# Dataset to load condensed images
class CondensedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load all image paths and their corresponding labels
        for label in range(num_classes):
            class_dir = os.path.join(root_dir, str(label))
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")  # Open image as RGB
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Real MNIST Test Dataset
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
real_mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size_test, shuffle=False)

# Execution
if __name__ == "__main__":
    # Generate and save condensed images
    generate_condensed_images_PAD(root_dir='./PAD_synthesis', visualize=True)

    # Load the synthesized condensed images dataset
    condensed_dataset = CondensedImageDataset(root_dir='./PAD_synthesis', transform=transform)
    condensed_loader = DataLoader(condensed_dataset, batch_size=batch_size_train, shuffle=True)

    # Define ConvNet3 model
    model = ConvNet(channel=1, num_classes=num_classes, net_width=128, net_depth=3,
                    net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train ConvNet3 on condensed images
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in condensed_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(condensed_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    end_time = time.time()
    train_time = end_time - start_time

    print("Training on condensed dataset completed.")
    print(f"Training Time on Real MHIST Data: {train_time:.2f} seconds")


    # Evaluate on real MNIST test data
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in real_mnist_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy on Real MNIST Data: {test_accuracy:.2f}%")
