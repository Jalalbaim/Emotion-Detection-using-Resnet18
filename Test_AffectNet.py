## Affect net 
## 3.11.3 anaconda/python.exe
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns


##from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load images
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Load annotations
        annotation_name = os.path.join(self.annotation_dir, self.images[idx].replace('.jpg', '_exp.npy'))
        annotation = int(np.load(annotation_name))
        
        if self.transform:
            image = self.transform(image)

        return image, annotation
    
class AffectClassifier:
    def __init__(self, dataset_path, batch_size=200):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.train_loader = None
        self.test_loader = None
        self.classes = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']


    def load_data(self):
       
        train_image_dir = os.path.join(self.dataset_path, 'train_set/train_set/images/')
        train_annotation_dir = os.path.join(self.dataset_path, 'train_set/train_set/annotations/')
        
        test_image_dir = os.path.join(self.dataset_path, 'val_set/val_set/images/')
        test_annotation_dir = os.path.join(self.dataset_path, 'val_set/val_set/annotations/')

        # Initialize the custom dataset for training and testing
        train_dataset = CustomDataset(image_dir=train_image_dir,
                                      annotation_dir=train_annotation_dir,
                                      transform=self.transform)

        test_dataset = CustomDataset(image_dir=test_image_dir,
                                     annotation_dir=test_annotation_dir,
                                     transform=self.transform)

        # Creating the data loaders
        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.batch_size, 
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=self.batch_size, 
                                                       shuffle=False)

    def visualize_data(self, num_images=9):
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        images = images[:num_images]
        labels = labels[:num_images]

        fig, axes = plt.subplots(1, num_images, figsize=(10, 2.5))
        for ax, image, label in zip(axes, images, labels):
            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        plt.show()

    def load_model(self):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 8)
        # pass to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("model loaded successfully")

    def save_checkpoint(self, state, filename="checkpoint.pth"):
        torch.save(state, filename)

    def load_checkpoint(self, filename="checkpoint.pth"):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}'".format(filename))
        else:
            print("No checkpoint found at '{}'".format(filename))
        

    def train_model(self, num_epochs=10):
        train_losses, val_losses, accuracies = [], [], []

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss, total_val_loss, total_correct, total_samples = 0, 0, 0, 0

            # Training phase
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Training Loss: {total_train_loss/5:.4f}")

            # Calculate and store average training loss
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                for images, labels in self.test_loader:
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()* images.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            # Calculate and store average validation loss
            avg_val_loss = total_val_loss / len(self.test_loader)
            val_losses.append(avg_val_loss)
            accuracy = 100 * total_correct / total_samples
            accuracies.append(accuracy)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Learning rate scheduling
            scheduler.step()

            # Saving checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': accuracy
            }
            self.save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{epoch+1}.pth")

        return train_losses, val_losses, accuracies


    def plot_loss_curve_from_checkpoints(self, checkpoint_dir="./checkpoints"):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
        all_train_losses, all_val_losses = [], []

        for file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, file)
            checkpoint = torch.load(checkpoint_path)

            all_train_losses.append(checkpoint['train_loss'])
            all_val_losses.append(checkpoint['val_loss'])
            print(f"Loaded checkpoint '{file}' with Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(all_train_losses, label='Training Loss')
        plt.plot(all_val_losses, label='Validation Loss')
        plt.title('Loss Curve from Checkpoints')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

    def load_checkpoint_and_evaluate(self, checkpoint_dir="./checkpoints2"):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
        all_train_losses, all_val_losses, all_accuracies = [], [], []

        for file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, file)
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['state_dict'])
            all_train_losses.append(checkpoint['train_loss'])
            all_val_losses.append(checkpoint['val_loss'])
            all_accuracies.append(checkpoint['accuracy'])
            print(f"Loaded checkpoint '{file}' with accuracy: {checkpoint['accuracy']:.2f}%")

        self.evaluate_model()
        return all_train_losses, all_val_losses, all_accuracies

    

    def display_predictions(self, num_images=6):
        self.model.eval()  # Set the model to evaluation mode

        # Get a batch of data from the test loader
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)

        # Select the number of images to display
        images, labels = images[:num_images], labels[:num_images]

        # Move the images to GPU if available
        if torch.cuda.is_available():
            images = images.cuda()

        # Inference
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)

        # Define class names (modify as per your dataset)
        classes = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']

        # Visualization
        fig, axs = plt.subplots(1, num_images, figsize=(12, 4))
        for i in range(num_images):
            image = images[i].cpu().numpy().transpose((1, 2, 0))  # Transpose and bring the image to CPU
            image = np.clip(image, 0, 1)  # Normalize the image to be between 0 and 1
            label = classes[predicted[i]]  # Get the predicted label

            axs[i].imshow(image)  # Show the image
            axs[i].set_title(f'Pred: {label}\nTrue: {classes[labels[i]]}')  # Show the predicted and true label
            axs[i].axis('off')  # Turn off the axis

        plt.show()
    
    def plot_confusion_matrix(self):
        # Set the model to evaluation mode
        self.model.eval()
        
        all_labels = []
        all_preds = []
        
        # No gradient needed
        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()
                
                # Get model predictions
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                # Move preds and labels to CPU, collect them
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.show()


    def calculate_f1_scores(self):
        self.model.eval()  # Set the model to evaluation mode

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()

                # Get model predictions
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                # Move preds and labels to CPU and collect them
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate F1 scores
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        f1_none = f1_score(all_labels, all_preds, average=None)  # F1 score for each class

        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print("F1 Score for each class:", f1_none)

        # Optionally, return the scores if you need to use them later
        return f1_macro, f1_micro, f1_weighted, f1_none
    
    # Classification report
    def generate_classification_report(self):
        self.model.eval()  # Set the model to evaluation mode

        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()
                
                # Get model predictions
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                # Move preds and labels to CPU and collect them
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Generate classification report
        report = classification_report(all_labels, all_preds, target_names=self.classes)
        print(report)
    
    def calculate_test_accuracy(self):
        classifier.model.eval()  # Set the model to evaluation mode

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                # Get model predictions
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                # Update total and correct predictions
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        # Calculate test accuracy
        accuracy = correct / total
        return accuracy



# Set the path to the dataset
path = 'C:/Users/JALAL/OneDrive/Bureau/BENet/transfer_6369964_files_84f3b5df/'

# Check if CUDA (GPU support) is available
print("CUDA is Available:", torch.cuda.is_available())

# Create an instance of AffectClassifier
classifier = AffectClassifier(dataset_path=path)

# Load the data
classifier.load_data()

# Visualize a batch of the data
classifier.visualize_data()

# Load the model
classifier.load_model()

# Start training the model
print('Model training started')
#train_losses, val_losses, accuracies = classifier.train_model(num_epochs=10)

# Plot the loss curve using the data from checkpoints
#classifier.plot_loss_curve_from_checkpoints()

checkpoint_dir = './checkpoints2'
checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")])
all_train_losses, all_val_losses = [], []

for file in checkpoint_files:
    checkpoint_path = os.path.join(checkpoint_dir, file)
    checkpoint = torch.load(checkpoint_path)

    all_train_losses.append(checkpoint['train_loss'])
    all_val_losses.append(checkpoint['val_loss'])
    print(f"Loaded checkpoint '{file}' with Train Loss: {checkpoint['train_loss']:.4f}, Val Loss: {checkpoint['val_loss']:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(all_train_losses, label='Training Loss')
plt.plot(all_val_losses, label='Validation Loss')
plt.title('Loss Curve from Checkpoints')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load the last checkpoint and evaluate the model
train_losses, val_losses, accuracies = classifier.load_checkpoint_and_evaluate()

# Evaluate the model's performance
classifier.evaluate_model()

# Display predictions from the model
classifier.display_predictions()

# Plot the confusion matrix
classifier.plot_confusion_matrix()

# Calculate F1 scores
classifier.calculate_f1_scores()

# Generate classification report
classifier.generate_classification_report()

# Calculate test accuracy
accuracy = classifier.calculate_test_accuracy()
print(f"Test Accuracy: {accuracy:.2f}%")