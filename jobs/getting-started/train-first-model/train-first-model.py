"""
The purpose of this script is to introduce AI Training.
We will show you how you can train your model by interacting with your own data, loaded either directly by the
code (thanks to some libraries), or by storing them in an OVH Object Storage. This Object Storage can then be added to
the environment of your job, which means that your model will have access to your data.
We will also show you how to save your model in the Cloud in order to retrieve it, once trained.
"""

# Step 1 - Set up environment
# Import libraries (Depending on the model you want to train, libraries will be different)
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zipfile

# Make sure the folder_name matches the mount directory you have specified for your Object Container / data
# Here we will go with my_data
folder_name = "my_data/"

# Initialize hyperparameters for the A.I. Network
nb_epochs = 5
batch_size = 64  # Using minibatches of 64 images
learning_rate = 0.001

# Step 2 - Load your data (here we use the FashionMNIST dataset)
"""Fashion-MNIST description
Dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples
Each sample is a 28×28 grayscale image which has an associated label from one of 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
"""

# Unzip the data file
with zipfile.ZipFile(folder_name+"my-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall(folder_name)

# Load the data which is in our Object Storage (you can also specify the complete path which is /workspace/folder_name)
train_csv = pd.read_csv(folder_name + "fashion-mnist_train.csv")
test_csv = pd.read_csv(folder_name + "fashion-mnist_test.csv")

# Name the classes of the dataset (10 possibilities)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Image transformations
# Convert the PIL images into Pytorch tensors + data normalization
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])


# Create a class to extract the data from the csv files
class FashionDataset:
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label = []
        image = []

        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])

        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


# Load the data dataset for the training/test sets (images + labels)
train_dataset = FashionDataset(train_csv, transform=transform)
test_dataset = FashionDataset(test_csv, transform=transform)

# Split the training set into training (80% size) and validation datasets (20% size)
train_size = int(len(train_dataset)*0.8)
val_size = len(train_dataset)-train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Wrap iterables around each set - Shuffle the data and divide it into different batches, using the maximum CPU threads (to make it fast)
num_cpu = mp.cpu_count()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, num_workers=2,shuffle=True) # reshuffles the data at every epoch to make sure we aren’t exposing our model to the same cycle of data in every epoch (to ensure the model isn’t adapting its learning to a pattern)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=2,shuffle=False) # Shuffle disabled since the order of samples won’t change the results
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=2,shuffle=False)

# Display lengths
#print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
#print("The validation set contains {} images, in {} batches".format(len(val_loader.dataset), len(val_loader)))
#print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

# Step 3 - Build the model (Classifier here)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)  # in_channels = 1 because grey scale image
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Create an instance of our Net class
model = Net()

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Step 4 - Prepare training
# Create an iterator that contains all our images (grouped by batch_size) so we can send them to our neural network
images, labels = next(iter(train_loader))

# Identify the current device type (CPU/CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("The model will be running on", device, "device")

# Convert model's parameters and buffers to CPU or Cuda
model.to(device)


# Define a function to save the model in our Object Storage, so we don't lose it when the job is finished
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Define a function to evaluate the model on the validation dataset
def test_accuracy(model, dataset, criterion):
    correct = 0.0
    total = 0
    running_loss = 0.0

    # evaluate model
    model.eval()
    with torch.no_grad():
        for data in dataset:
            inputs, labels = data

            # send images & labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            # run the model on the test set to predict labels
            outputs = model(inputs)

            # compute the loss
            loss = criterion(outputs, labels)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()

            # compute the accuracy over all test images
    val_accuracy = correct / total
    val_loss = running_loss

    # back to train mode
    model.train()

    return round(val_accuracy, 4), round(val_loss, 4)


# Define a function that returns the number of correct predictions (preds compared to labels)
def compare_preds_labels(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# Step 5 - Train the neural network
# Initialize some variables
best_accuracy = 0.0
train_losses = []
val_losses = []

# Random control
torch.manual_seed(123)

# Loop over the dataset multiple times
for epoch in range(nb_epochs):
    total_correct = 0
    total_loss = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # send images & labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(inputs)

        # Compute the Loss (how wrong our results are)
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Optimize & Update Weights
        optimizer.step()

        # Calculate Loss
        # running_loss += loss.item()
        total_loss += loss.item()

        # losses.append(loss.data);
        total_correct += compare_preds_labels(outputs, labels)
        accuracy = total_correct / len(train_dataset)

    total_loss = round(total_loss, 4) / 1000

    # Compute and print the obtained accuracy for this epoch when tested over all our validation images
    val_accuracy, val_loss = test_accuracy(model, val_loader, criterion)

    # Save the model each time the val_accuracy is improved
    if val_accuracy > best_accuracy:
        save_model(model, path=folder_name + "model.net")
        best_accuracy = val_accuracy

    print('Epoch : %d/%d,  Loss: %f, Accuracy: %.3f, Val_loss: %f, Val_accuracy: %f' % (epoch + 1, nb_epochs, total_loss, accuracy, val_loss, val_accuracy))
    train_losses.append(round(total_loss, 3))
    val_losses.append(round(val_loss, 4))

print('\n--- Finished Training ---\n')

# Optionnal - Display losses
plot = False
if plot:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Valid loss")
    plt.legend()
    plt.show()

# Step 6 - Test the model for each class

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in test_loader:
        inputs, labels = data

        # send images & labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate outputs by running images through the network
        outputs = model(inputs)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
print("\n\n--- Results of the model on the test set ---\n")
total_accuracy = 0
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    total_accuracy += accuracy

print('Mean accuracy of the model on the whole test set: %.1f %%' % (total_accuracy / len(classes)))