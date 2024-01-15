# Main Class
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import matplotlib.pyplot as plt
from Test_AffectNet import AffectClassifier


path = 'C:/Users/JALAL/OneDrive/Bureau/BENet/transfer_6369964_files_84f3b5df/'

# Check if CUDA (GPU support) is available
print("CUDA is Available:", torch.cuda.is_available())

# Classifier object
classifier = AffectClassifier(dataset_path=path)

# Load the datas
classifier.load_data()

# Visualize a batch of the data
classifier.visualize_data()

# Load the model
classifier.load_model()
print(classifier.model)

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
#accuracy = classifier.calculate_test_accuracy()
#print(f"Test Accuracy: {accuracy:.2f}%")