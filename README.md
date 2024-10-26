# Image Classifier Using PyTorch

---

### Project Overview
The objective of this project was to build a convolutional neural network (CNN)-based image classifier using PyTorch, trained on the MNIST dataset of handwritten digits (0-9). By leveraging CNN's spatial hierarchy-capturing capabilities, we aimed to build a model that accurately classifies images of digits.

---

### Learning Objectives

#### 1. Understanding Convolutional Neural Networks (CNNs)
   - Learned the basic structure and functions of CNNs, such as:
     - **Convolutional Layers**: for extracting spatial features
     - **Activation Functions**: such as ReLU for non-linearity
     - **Pooling Layers**: for dimensionality reduction
     - **Fully Connected Layers**: for final classification.
   - Insight into how CNNs excel at image classification by identifying hierarchical spatial patterns in images.

> _**Image:** ![image](https://github.com/user-attachments/assets/31d90f41-17ad-43d9-980b-af740559c6ed)


#### 2. Data Handling with PyTorch
   - **torchvision** was used to load and preprocess the MNIST dataset efficiently.
   - Understood image preprocessing techniques, including normalization and tensor conversion for compatibility with PyTorch models.

```python
import torch
from torchvision import datasets, transforms

# Data transformation and loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
```

#### 3. Model Training and Evaluation
   - Gained practical insights into the neural network training process:
     - **Forward Propagation**: computing predictions
     - **Loss Calculation**: using CrossEntropyLoss
     - **Backpropagation and Optimization**: leveraging the Adam optimizer.

```python
import torch.nn as nn
import torch.optim as optim

# Model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

> _**Image:** ![image](https://github.com/user-attachments/assets/e470db23-d498-42ce-ad0e-6ecf66da56eb)


#### 4. Device Management
   - Ensured device compatibility (CPU/GPU) to maximize performance and enable model training on available hardware.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

#### 5. Model Persistence
   - Practiced model saving and loading for deployment using PyTorch’s `torch.save()` and `torch.load()` functions.

```python
# Save model
torch.save(model.state_dict(), 'mnist_cnn.pth')

# Load model
model.load_state_dict(torch.load('mnist_cnn.pth'))
```

---

### Challenges Encountered

#### 1. CUDA Compatibility
   - Faced initial issues with GPU support, requiring code modifications to run on the CPU when necessary.

```python
# Check for CUDA
if torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()
```

#### 2. Data Loading and Preprocessing
   - Ensuring correct image transformations and tensor compatibility with the model architecture was challenging but critical for effective training.

#### 3. Hyperparameter Tuning
   - Experimented with learning rates, batch sizes, and optimizer settings. Tuning these hyperparameters significantly impacted model performance.

#### 4. Model Overfitting
   - Observed overfitting during training, particularly with a high training accuracy and lower validation accuracy. Addressed this by incorporating dropout layers and regularization techniques.

#### 5. Debugging and Error Handling
   - Encountered tensor shape mismatches and model loading issues, which required a deeper understanding of the PyTorch framework and its debugging methods.

---

### Results

- Achieved an accuracy of **98%** on the MNIST test set after training for 10 epochs.
- The model demonstrated strong generalization on unseen data, indicating effective learning.

```python
# Example testing accuracy calculation
correct, total = 0, 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
```

> _**Image:** ![image](https://github.com/user-attachments/assets/ec2c1725-c462-4e9a-a2a0-32b6a684247d)  ![image](https://github.com/user-attachments/assets/e99c3fc6-24bb-4cd5-a756-67bbda363512)



---

### Conclusion
This project provided hands-on experience in building an image classifier using PyTorch and CNNs. The workflow, from loading data to training and evaluating the model, solidified fundamental machine learning concepts. The challenges faced enriched problem-solving skills and deepened understanding of neural network training.

>
