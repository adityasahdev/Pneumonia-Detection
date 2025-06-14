import os 
from PIL import Image
import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader 


transform = transforms.Compose([        #create pipeline
    transforms.Resize((224,224)),       #resize since neural networks expect fixed size inputs 
    transforms.Grayscale(),  #Ensure single channel since chest Xray is grayscale(white and black) nd CNNs operate in 3 channels (RGB), if its not changed CNN treat Xray as 3 channels wasting resources
    transforms.ToTensor(),      # converts image into pytorch tensor, sclaes pixel from [0,255] to [0.0,1.0] for model stability and model convergence 
    transforms.Normalize(mean = [0.5], std = [0.5])     # normlizes pixel values from [0,1] to [1,1], neural networks train better when data is near 0
])


train_dataset_path = "/Users/adityaahdev/Desktop/Dataset Projects/Pneumonia Detection /chest_xray/train"
validate_dataset_path = "/Users/adityaahdev/Desktop/Dataset Projects/Pneumonia Detection /chest_xray/val"
test_dataset_path = "/Users/adityaahdev/Desktop/Dataset Projects/Pneumonia Detection /chest_xray/test"

train_dataset = datasets.ImageFolder(train_dataset_path,transform=transform) #applies transforms and gathers label folders inside train folder 
val_dataset = datasets.ImageFolder(validate_dataset_path,transform=transform)
test_dataset = datasets.ImageFolder(test_dataset_path,transform=transform)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  #converts image shpae into [32(batch size),1(channels, 1 because greyscale),244(Resized H),244(Resized W)]


import torch.nn as nn 
import torch.nn.functional as F 

class PneumoniaCNN(nn.Module): 
    def __init__(self):
        super(PneumoniaCNN,self).__init__()

        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1) #[1,244,244] -> [16,244,244] #paddind keeps output size the same 
        self.pool = nn.MaxPool2d(2,2) #halves size -> [16,244,244] -> [16,122,122]

        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1) #[16,122,122] -> [32,122,122]
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1) #[32,122,122] -> [64,122,122]

        self.fc1 = nn.Linear(64*28*28,128)  #Flatten input -> (input size * after 3 conv layers 224->112->56->28 * 28 * 128(output) )
        self.fc2 = nn.Linear(128,2) # 2 output classes

        self.dropout = nn.Dropout(0.3)
    def forward(self,x): #defines the flow of input
        x = self.pool(F.relu(self.conv1(x))) #conv -> Relu -> Max pooling [1,244,244] -> [ 16,122,122]
        x = self.pool(F.relu(self.conv2(x))) #conv > Relu > pool [32,56,56]
        x = self.pool(F.relu(self.conv3(x))) #conv > Relu > pool [64,28,28]
        x = x.view(-1,64*28*28) #Flatten the tensor to feed into fully connected layers ( -1 means infer the batch size dynamically)
        x = self.dropout(F.relu(self.fc1(x)))   #dropout layer
        x = self.fc2(x) # give two output classes
        return x 

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = PneumoniaCNN().to(device)

import torch.optim as optim 

criterion =  nn.CrossEntropyLoss()      #because two classes
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 10 
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0 
    correct = 0
    total = 0

    for images , labels in train_loader:
        images,labels = images.to(device) , labels.to(device) 

        optimizer.zero_grad() #clear previous gradients 
        outputs = model(images)     #forward pass
        loss = criterion(outputs,labels)    #compute loss
        loss.backward()     #backpropogate 
        optimizer.step()    #update weights 

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct/total 
    print(f"Epoch [{epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")


model.eval()
test_loss= 0.0
test_correct = 0
test_total = 0 

with torch.no_grad():
    for images , labels in test_loader: 
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs,labels)
        test_loss += loss.item() 

        _, predicted = torch.max(outputs.data,1 )
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct/test_total 
test_loss = test_loss/len(test_loader)
print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")
