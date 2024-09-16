import torch

from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Define the structure of the neural network
        self.linear_relu_stack = nn.Sequential(
            # first layer dimension of the image (300*300)*3 RGB, 256 values for RGB
            nn.Linear(3 * 300 * 300, 256),
            nn.ReLU(),
            # 2 Layer with 4096 neurons
            nn.Linear(256, 4096),
            nn.ReLU(),
            # 3 Layer with 2048 neurons
            nn.Linear(4096, 2048),
            nn.ReLU(),
            # 4 Layer with 256
            nn.Linear(2048, 256),
            nn.ReLU(),
            # 5 Layer with 32
            nn.Linear(256, 32),
            nn.ReLU(),
            # 6 layer with the output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def main():

    # Define the transform to be applied to images, resize the image, convert to tensor and normalize
    transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load the dataset
    dataset = datasets.ImageFolder('.\dataset', transform=transform)


    # Shows the classes of the dataset
    # print(dataset.classes, end = '\n\n')
    # Shows class to index mapping
    # print(dataset.class_to_idx, end = '\n\n')
    # Print img and label of the first image
    # print(dataset[0], end = '\n\n')

    # Calculate train_set and test_set size with a 80% split
    train_set_size = int(len(dataset)*0.8)
    test_set_size = len(dataset) - train_set_size

    # Split the dataset in train and test sets
    train_set, test_set = random_split(dataset, lengths=[train_set_size, test_set_size])

    batch_size = 10
    # Creates dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Choose the device on which to run the training
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Initialize the model
    model = NeuralNetwork().to(device)

    # Print the model
    # print(model)
    
    # Calculate loss function with binary cross entropy as the output is binary
    loss_fn = nn.BCELoss()
    # Define the optimizer lr=learning re
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Activate training mode
    model.train()

    # Defines values for epochs 
    epochs = 1
    for epoch in range(epochs):
        for batch, (x, y) in enumerate(train_loader):
            # Retrieve features and labels in a batch specified when creating the dataloader
            x, y = x.to(device), y.to(device).float()
            y = y.unsqueeze(1) 
            # Print feature and label
            # print(f"Feature batch shape: {x}")
            # print(f"Labels batch shape: {y}")

            # Makes the prediction
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Batch = {batch}, Loss = {loss.item()}\n")
    
    # Saves the model
    model_path = r"C:\Users\ex1pontigi\OneDrive - PRYSMIAN GROUP\Documents\Documenti Personali\Artificial Intelligence\02 - Projects\02 - Neural Networks\almond_damage_recognition\model.pth"
    torch.save(model, model_path)

if __name__ == "__main__":
    main()
