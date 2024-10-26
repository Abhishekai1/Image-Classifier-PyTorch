# Import dependencies
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to(device)  # Use the device variable
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    for epoch in range(10):  # train for 10 epochs
        for batch in dataset: 
            X, y = batch 
            X, y = X.to(device), y.to(device)  # Use the device variable
            yhat = clf(X) 
            loss = loss_fn(yhat, y) 

            # Apply backprop 
            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch: {epoch} loss is {loss.item()}")
    
    # Save model state
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    # Load model state
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    # Load and preprocess image
    
    img = Image.open('C:\\Users\Amresh Yadav\Desktop\E2E\PyTorchin15-main\img_3.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # Use the device variable

    # Make prediction
    print(torch.argmax(clf(img_tensor)))