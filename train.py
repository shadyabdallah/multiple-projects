import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse

class Network(nn.Module):
    """ Defines the classifier's network architcture
        - Hidden_layers are allowed in as a list of integers, where the length of the list isthe number of layers, 
        and values are layer's width.
        - dropout is included (default at p = 0.5)
        - ReLU is used between layers
        - log_softmax is used on the final output
    """
    def __init__(self, input_size, output_size, hidden_layers, drop_p = 0.5):
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
    
        self.dropout = nn.Dropout(p = drop_p)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        x = self.output(x)
        return F.log_softmax(x, dim = 1)
    
def training(model, criterion, optimizer, trainloader, validloader, epochs):  
    """ A function used to train the feedforward classifier"""
    print_every = 20
    steps = 0
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            steps += 1
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion)
                print("Epoch {}/{}....".format(e+1, epochs),
                      "Training Loss: {:.3f}....".format(running_loss/print_every),
                      "Test Loss: {:.3f}....".format(test_loss/len(validloader)),
                      "Accuracy: {:.3f}....".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
    return model

def validation(model, validloader, criterion):
    """ a function used for validation, returns testloss and accuracy """
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in validloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model.forward(images)
        
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def save_checkpoint(model, train_datasets, filepath):
    """ saves a checkpoint of the model, takes in:
        - model archtitecture
        - class_to_idx of the datasets
        - state_dict of the model.
        - input size
        - output size
        - hidden layers
    """
    model.class_to_idx = train_datasets.class_to_idx
    checkpoint = {'model' : args.arch,
                 'class_to_idx' : model.class_to_idx,
                 'state_dict': model.state_dict(),
                 'input_size' : input_size,
                 'output_size' : output_size,
                 'hidden_layers' : args.hidden_layers}

    torch.save(checkpoint, filepath)

def arg():
    """ defines command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type = str, help = "directory in which the data is located")
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = "saving directory and file name for the checkpoint. Example: 'datadirectory/checkpoint.pth'")
    parser.add_argument('--arch', default = 'vgg16', choices = ['resnet18', 'alexnet', 'vgg16', 'vgg13', 'vgg16', 'densenet', 'googlenet', 'mobilenet'], help = "model architecture, various choices are available to select from. Example: vgg16")
    parser.add_argument('--gpu', type = str, default = 'cuda:0', help = "device to be used, either 'cpu' or 'cuda:0' as a default")
    parser.add_argument('--hidden_layers', nargs = "+", default = [4096], help = "Hidden layers filled in as a list of integers with spaces in between. Example 4096 1024 512.")
    parser.add_argument('--epochs', type = int, default = 5, help = "number of epochs for the model to iterate on, supply an integer")
    parser.add_argument('--learning_rate', type = float, default = 0.0001, help = "learning rate, fill in a floating point integer")
    args = parser.parse_args()
    return args

args = arg()

#device determination
if args.gpu == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      
#data directories, image transformations and dataloaders        
train_dir = args.data_directory + '/train'
valid_dir = args.data_directory + '/valid'
test_dir = args.data_directory + '/test'

#dataset transformations
train_transforms = transforms.Compose([transforms.Resize(250), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(30), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(250), transforms.CenterCrop(224), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#applying transformations and loading datasets
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 32, shuffle = True)

#model initialization
input_size = 25088
output_size = 102

model = eval("models."+ args.arch + "(pretrained = True)")
for param in model.parameters():
    param.requires_grad == False

model.classifier = Network(input_size, output_size, args.hidden_layers)
model.to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
criterion = nn.NLLLoss() 

#starting model training
training(model, criterion, optimizer, trainloader, validloader, args.epochs)

#additional testing on a batch of images from the testing dataset
model.eval()
with torch.no_grad():
    test_loss, accuracy = validation(model, testloader, criterion)
print("Additional Test Performed on Test Data: \nLoss {:.3f}....".format(test_loss/len(testloader)),
     "Accuracy {:.3f}....".format(accuracy/len(testloader)))

#saving the checkpoint
save_checkpoint(model, train_datasets, args.save_dir)