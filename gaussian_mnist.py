import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from networks import ConvNet  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    test_accuracy = correct / total
    print(f"Test Accuracy after Attention Matching with Gaussian Noise Initialization: {test_accuracy:.4f}")


# Define the attention matching function with synthetic data initialized from Gaussian noise
def attention_matching_mnist(
    K=100,
    model_steps=50,           
    lr_synthetic=0.1,
    synthetic_steps=1,
    lr_model=0.01,
    lambda_task=0.01,
    iteration=10,
    images_per_class=10,
    minibatch_size=256
):
    # Set up MNIST data loader
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(mnist_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=minibatch_size, shuffle=False)

    num_classes = 10
    im_size = (1, 28, 28)  
    synthetic_data = [
        torch.randn((images_per_class, *im_size), device=device, requires_grad=True) 
        for _ in range(num_classes)
    ]

    synthetic_labels = torch.tensor([i for i in range(num_classes) for _ in range(images_per_class)], device=device)

    # Start Attention Matching process
    for itr in range(iteration):
        print(f"Epoch {itr+1}/{iteration}")
        epoch_loss = 0  

        for weight_init_iter in range(K):
            model = ConvNet(
                channel=1, num_classes=10, net_width=128, net_depth=3,
                net_act='relu', net_norm='batchnorm', net_pooling='avgpooling', im_size=(28, 28)
            ).to(device)

            model_optimizer = optim.SGD(model.parameters(), lr=lr_model, momentum=0.9)

            # Model training on synthetic data for `model_update_steps` iterations
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

        print(f"Iteration {itr+1} Completed - Average Attention Loss: {epoch_loss / (K * model_steps):.4f}")

    visualize_synthetic_images(synthetic_data)

    evaluate_model(model, test_loader)

def visualize_synthetic_images(synthetic_data):
    filename = "MNIST_Guassian.png"
    num_classes = len(synthetic_data)
    images_per_class = synthetic_data[0].shape[0]  

    fig, axes = plt.subplots(num_classes, images_per_class, figsize=(12, 12))
    for class_idx, class_data in enumerate(synthetic_data):
        for img_idx in range(images_per_class):
            image = class_data[img_idx].detach().cpu().squeeze()
            axes[class_idx, img_idx].imshow(image, cmap='gray')
            axes[class_idx, img_idx].axis("off")
            if img_idx == 0:
                axes[class_idx, img_idx].set_title(f"Class {class_idx}")

    plt.suptitle("Condensed Synthetic Images per Class Initialized with Gaussian Noise")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Run the Attention Matching training with Gaussian noise initialization
attention_matching_mnist()

