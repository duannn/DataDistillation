import os
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import random
from PIL import Image
from networks import ConvNet 
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the attention matching function 
def attention_matching_mnist(K, model_steps, lr_synthetic, synthetic_steps, lr_model, lambda_task, iteration, images_per_class, minibatch_size, folder_path):
    # Set up MNIST data loader
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=minibatch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=minibatch_size, shuffle=False)
    num_classes = 10
    synthetic_data = []

    mnist_images_by_class = {i: [] for i in range(num_classes)}
    for image, label in mnist_train:
        mnist_images_by_class[label].append(image)

    # Randomly select images_per_class real images for each class
    for class_idx in range(num_classes):
        selected_images = random.sample(mnist_images_by_class[class_idx], images_per_class)
        class_tensor = torch.stack([img for img in selected_images]).to(device).requires_grad_(True)
        synthetic_data.append(class_tensor)

    synthetic_labels = torch.tensor([i for i in range(num_classes) for _ in range(images_per_class)], device=device)

    # Start Attention Matching process
    for itr in range(iteration):
        print(f"Epoch {itr+1}/{iteration}")
        average_epoch_loss = 0

        for weight_init_iter in range(K):
            model = ConvNet(
                channel=1, num_classes=10, net_width=128, net_depth=3,
                net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)
            ).to(device)

            model_optimizer = optim.SGD(model.parameters(), lr=lr_model, momentum=0.9)

            for update_step in range(model_steps):
                synthetic_inputs = torch.cat([sd.clone().detach().requires_grad_(True) for sd in synthetic_data]).to(device)
                model_output = model(synthetic_inputs)
                model_loss = nn.CrossEntropyLoss()(model_output, synthetic_labels)

                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()
                average_epoch_loss += model_loss.item()

            for synthetic_step in range(synthetic_steps):
                for class_idx, synthetic_class_data in enumerate(synthetic_data):
                    synthetic_class_data = synthetic_class_data.clone().detach().requires_grad_(True)
                    synthetic_optimizer = optim.SGD([synthetic_class_data], lr=lr_synthetic)

                    real_images, real_labels = next(iter(train_loader))
                    real_images, real_labels = real_images.to(device), real_labels.to(device)

                    model.eval()
                    synthetic_output = model(synthetic_class_data)
                    real_output = model(real_images)

                    attention_loss = lambda_task * ((synthetic_output - real_output[:images_per_class].detach()) ** 2).mean()

                    synthetic_optimizer.zero_grad()
                    attention_loss.backward()
                    synthetic_optimizer.step()

                    synthetic_data[class_idx] = synthetic_class_data.detach().requires_grad_(True)

        print(f"Iteration {itr+1} Completed - Average Attention Loss: {average_epoch_loss / (K * model_steps):.4f}")

    # Save synthetic images to folder
    save_synthetic_images(synthetic_data, folder_path)
    return test_loader

# Save each synthesized image as a PNG file in the specified folder
def save_synthetic_images(synthetic_data, save_path):
    os.makedirs(save_path, exist_ok=True)
    for class_idx, class_data in enumerate(synthetic_data):
        class_folder = os.path.join(save_path, str(class_idx))
        os.makedirs(class_folder, exist_ok=True)
        for i, img_tensor in enumerate(class_data):
            img = transforms.ToPILImage()(img_tensor.detach().cpu())
            img.save(os.path.join(class_folder, f"synthetic_img_{i}.png"))

# Load synthetic images from folder as a dataset
class SyntheticMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for class_label in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_label)
            for img_file in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, img_file))
                self.labels.append(int(class_label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # MNIST is grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Training function on synthetic dataset 
def train_on_synthetic_folder_data(synthetic_folder_path, test_loader, num_epochs=20):
    transform = transforms.Compose([transforms.ToTensor()])
    synthetic_dataset = SyntheticMNISTDataset(root_dir=synthetic_folder_path, transform=transform)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=16, shuffle=True)

    model = ConvNet(channel=1, num_classes=10, net_width=128, net_depth=3,
                    net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in synthetic_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        print(f"Iteration {epoch+1}/{num_epochs}, Loss: {running_loss / len(synthetic_loader.dataset):.4f}")

    end_time = time.time()
    train_time = end_time - start_time

    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy on Synthetic Data: {test_accuracy:.4f}")
    print(f"Training Time on Synthetic Data: {train_time:.2f} seconds")
    return test_accuracy, train_time

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Run the Attention Matching training with specified parameters and save images to folder
test_loader = attention_matching_mnist(
    K=100, model_steps=50, lr_synthetic=0.1, synthetic_steps=1, 
    lr_model=0.01, lambda_task=0.01, iteration=10, images_per_class=10,
    minibatch_size=256, folder_path='./mnist_synthesis'
)

# Train ConvNet3 on the synthetic images saved in folder and test it
test_accuracy, train_time = train_on_synthetic_folder_data('./mnist_synthesis', test_loader)


def calculate_flops(model, input_size):
    model_summary = summary(model, input_size=input_size, verbose=0)
    flops = model_summary.total_mult_adds  # Total multiply-adds
    return flops

# Evaluation function 
def evaluate_model_with_flops_on_test_set(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_flops = 0  # Accumulate FLOPs across batches

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            flops = calculate_flops(model, input_size=(batch_size, 1, 28, 28))
            total_flops += flops

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Average FLOPs per sample across the test set
    avg_flops = total_flops / total
    accuracy = correct / total
    print(f"Average FLOPs per sample across test set: {avg_flops:.2e}")
    return accuracy, avg_flops

# Training function for real MNIST dataset with FLOPs calculation on test set
def train_on_real_mnist_with_flops(num_epochs=20):
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

    # Initialize ConvNet3 model
    model = ConvNet(channel=1, num_classes=10, net_width=128, net_depth=3,
                    net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Measure training time
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}")

    end_time = time.time()
    train_time = end_time - start_time

    # Evaluate the model on the test set and calculate FLOPs
    test_accuracy, avg_flops = evaluate_model_with_flops_on_test_set(model, test_loader)
    print(f"Training Time on Real MNIST Data: {train_time:.2f} seconds")
    print(f"Test Accuracy on Real MNIST Data: {test_accuracy:.4f}")
    print(f"Average FLOPs per sample on Test Set: {avg_flops:.2e}")
    return test_accuracy, train_time, avg_flops

# Run training and evaluation on real MNIST dataset 
real_test_accuracy, real_train_time, avg_test_flops = train_on_real_mnist_with_flops(num_epochs=20)

print("\nComparison of ConvNet3 on Real vs. Synthetic MNIST Data")
print(f"Real MNIST - Test Accuracy: {real_test_accuracy:.4f}, Training Time: {real_train_time:.2f} seconds")
print(f"Synthetic MNIST - Test Accuracy: {test_accuracy:.4f}, Training Time: {train_time:.2f} seconds")


def visualize_synthetic_images(synthetic_dir, images_per_class=10, num_classes=10):
    filename = "MNIST_Synthesis.png"
    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(images_per_class, num_classes))

    for class_idx in range(num_classes):
        for img_idx in range(images_per_class):
            img_path = os.path.join(synthetic_dir, str(class_idx), f"synthetic_img_{img_idx}.png")
            image = Image.open(img_path).convert('L')  # Ensure grayscale mode
            axes[class_idx, img_idx].imshow(image, cmap='gray')
            axes[class_idx, img_idx].axis("off")

    plt.suptitle("Synthesized MNIST Images", fontsize=16)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Run the visualization
visualize_synthetic_images('./mnist_synthesis')
