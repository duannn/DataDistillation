import os
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from networks import ConvNet 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_accuracy = correct / total
    print(f"Test Accuracy after Attention Matching with Gaussian Noise Initialization: {test_accuracy:.4f}")

# Define MHISTDataset with label mapping
class MHISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_path, partition="train", transform=None):
        self.root = root
        self.transform = transform
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data['Partition'] == partition]
        
        # Map labels from 'SSA'/'HP' to integers (e.g., SSA -> 0, HP -> 1)
        label_mapping = {"SSA": 0, "HP": 1}
        self.labels = self.data['Majority Vote Label'].map(label_mapping).values  # Apply mapping
        self.image_paths = self.data['Image Name'].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Define the attention matching function with synthetic data initialized from Gaussian noise
def run_attention_matching_mhist(
    K=20,
    model_steps=50,
    lr_synthetic=0.15,
    synthetic_steps=1,
    lr_model=0.015,
    lambda_task=0.01,
    iteration=10,
    images_per_class=50,
    minibatch_size=128
):
    # Load MHIST dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = './mhist_dataset/mhist_images'  
    csv_path = './mhist_dataset/annotations.csv' 
    mhist_dataset = MHISTDataset(root=data_path, csv_path=csv_path, partition="train", transform=transform)
    real_loader = DataLoader(mhist_dataset, batch_size=minibatch_size, shuffle=True)

    # Initialize synthetic data with Gaussian noise for each class
    num_classes = 2  # Assuming two classes for MHIST: HP and SSA
    im_size = (3, 128, 128)  # Image dimensions for MHIST
    synthetic_data = [
        torch.randn((images_per_class, *im_size), device=device, requires_grad=True, dtype=torch.float32) 
        for _ in range(num_classes)
    ]

    # Generate labels for the synthetic dataset
    synthetic_labels = torch.tensor([i for i in range(num_classes) for _ in range(images_per_class)], device=device, dtype=torch.long)

    # Start Attention Matching process
    for itr in range(iteration):
        print(f"Epoch {itr+1}/{iteration}")
        epoch_loss = 0  # Track loss for each epoch

        for weight_init_iter in range(K):
            model = ConvNet(
                channel=3, num_classes=num_classes, net_width=128, net_depth=7,
                net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(128, 128)
            ).to(device)

            model_optimizer = optim.SGD(model.parameters(), lr=lr_model, momentum=0.9)

            for update_step in range(model_steps):
                synthetic_inputs = torch.cat([sd.clone().detach().requires_grad_(True) for sd in synthetic_data]).to(device)

                model_output = model(synthetic_inputs)
                model_loss = nn.CrossEntropyLoss()(model_output, synthetic_labels)

                model_optimizer.zero_grad()
                model_loss.backward()
                model_optimizer.step()
                epoch_loss += model_loss.item()

            # Update synthetic data with attention matching
            for synthetic_step in range(synthetic_steps):
                for class_idx, synthetic_class_data in enumerate(synthetic_data):
                    synthetic_class_data = synthetic_class_data.clone().detach().requires_grad_(True)
                    synthetic_optimizer = optim.SGD([synthetic_class_data], lr=lr_synthetic)

                    real_images, real_labels = next(iter(real_loader))
                    real_images, real_labels = real_images.to(device), real_labels.to(device)

                    model.eval()
                    synthetic_output = model(synthetic_class_data)
                    real_output = model(real_images)

                    attention_loss = lambda_task * ((synthetic_output - real_output[:images_per_class].detach()) ** 2).mean()

                    synthetic_optimizer.zero_grad()
                    attention_loss.backward()
                    synthetic_optimizer.step()

                    synthetic_data[class_idx] = synthetic_class_data.detach().requires_grad_(True)

        print(f"Iteration {itr+1} Completed - Average Attention Loss: {epoch_loss / (K * model_steps):.4f}")

    visualize_synthetic_images(synthetic_data)

    evaluate_model(model, real_loader)


def visualize_synthetic_images(synthetic_data):
    filename = "MHIST_Gaussian.png"
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle("MHIST Synthetic Images Initialized with Gaussian Noise", fontsize=16)

    img_idx = 0
    for class_idx, class_data in enumerate(synthetic_data):
        for i in range(50):  # 50 images per class
            row, col = divmod(img_idx, 10)
            image = class_data[i].detach().cpu().permute(1, 2, 0)  # Convert to (H, W, C)
            axes[row, col].imshow(image, vmin=0, vmax=1)
            axes[row, col].axis("off")
            img_idx += 1

    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Run the Attention Matching training with Gaussian noise initialization
run_attention_matching_mhist()
