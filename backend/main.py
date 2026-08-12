from io import BytesIO
import torch
from torch import torch_version
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
# Inherit the base Neural net class to make the new Net class



class Net(nn.Module):
    # Initialize key properties of the neural network
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layer with 3 input channels and 6 output channels , 5 Defines the size of the sliding window
        self.conv1 = nn.Conv2d(3, 6, 5) # Input layer
        
        self.pool = nn.MaxPool2d(2, 2)
        # Pooling layer with a kernel size of 2*2 and stride of 2

        self.conv2 = nn.Conv2d(6, 16, 5) # Convolutional layer with 6 different channels
        # Takes 6 input channels, produces 16 output channels
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Applies a linear tranformation to incoming data
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 -> The final number of classes
        # Output layer

    # Defines the direction of data flow from input to ouput
    # (Forward pass implementation)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Input layer
        x = self.pool(F.relu(self.conv2(x))) # Hidden layer
        x = x.view(-1, 16 * 5 * 5) # Flattening is necessary because our dense layer expects 1-D data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Why are we not doing relu here
        return x
    

MODEL_PATH =  Path(__file__).resolve().parent / "cifar_model.pth"
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

net = Net()
if MODEL_PATH.exists():
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()

    print(f"Model loaded from {MODEL_PATH}")
else:
    print("NO SUCH MODEL")


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Define the list of allowed frontend origins
origins = [
    "*"
]

# Add the CORS middleware to your application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # Enforces the specified domains list
    allow_credentials=True,        # Permits cookies and authentication headers
    allow_methods=["*"],           # Allows all standard HTTP methods
    allow_headers=["*"],           # Allows all request headers
)

@app.get("/")
def read_root():
    return {"message": "CORS configuration is active!"}




#API NDIO HII HAPA

app = FastAPI(title="CIFAR-10")

@app.get('/')
def root():
    return {"message": "Welcome to the CIFAR-10 Prediction API"}

@app.post('/predict')
def predict(file: UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Please upload a valid image")
    try:
        contents = file.file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')

    except Exception:
        raise HTTPException(status_code=400, detail="Couldn't load file")
    image = transform(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        outputs = net(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {"prediction": classes[predicted.item()], "confidence": confidence.item()}










