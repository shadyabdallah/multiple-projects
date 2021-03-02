import torch 
from torchvision import models
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import json

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
def arg():
    """ defines command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type = str, help = "Path of the image to be test. Example: 'ImageClassifier/image.jpeg")
    parser.add_argument('checkpoint', type = str, help = "checkpoint path to be loaded. Example: 'ImageClassifier/checkpoint.pth ")
    parser.add_argument('--gpu', type = str, default = 'cuda:0', help = "device to be used, either 'cpu' or 'cuda:0' as a default")
    parser.add_argument('--topk', type = int, default = 3, help = "number of top probabilities and categories to be presented")
    parser.add_argument('--category_names', type = str, default = 'ImageClassifier/cat_to_name.json', help = "path of category-to-name file")
    
    args = parser.parse_args()
    return args

def load_checkpoint(filepath):
    """ loads checkpoint and returns the model with all the required parameters"""
    
    checkpoint = torch.load(filepath)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    modeltype = checkpoint['model']
    model = eval("models."+ modeltype + "(pretrained = True)")
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.classifier = Network(input_size, output_size, hidden_layers)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a pytorch tensor.
    '''
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image.resize((256, 256)).crop((16, 16, 240, 240))
    np_image = np.array(img) / 255
    np_image = (np_image - mean)/std 
    proc_image = torch.from_numpy(np_image.transpose(2,0,1)).type(torch.FloatTensor)
    
    return proc_image

def predict(image_path, model, cat_to_name, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(Image.open(image_path))
    image = image.unsqueeze_(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    props, idx = ps.topk(topk)
    classes = []
    for i in idx.squeeze():
        for clss, indx in model.class_to_idx.items():
            if indx == i:
                classes.append(clss)
                
    names = []
    for i in classes:
        for cat, name in cat_to_name.items():
            if cat == i:
                names.append(name)
                
    return names, classes, props

def sanity_check(image_path, model, cat_to_name):
    """ performs a sanity check on the image and prediction generated
    currently not used since the script is run on terminal.
    """
    #imported locally within the function since it's not used in this script
    import pandas as pd
    import seaborn as sb
    import matplotlib.pyplot as plt
    
    img = Image.open(image_path)
    classes, props = predict(image_path, model, args.topk)
    names = []
    for i in classes:
        for cat, name in cat_to_name.items():
            if cat == i:
                names.append(name)

    df = pd.DataFrame({'Names' : names, 'Probabilities' : props.squeeze()})

    fig, ax = plt.subplots(2,1, figsize = (5,8))
    ax[0].imshow(img)
    ax[0].set_title(names[0])
    ax[0].axis('off')
    sb.barplot(df['Probabilities'], df['Names'], order = df['Names'], color = 'grey');


args = arg()
#device selection
if args.gpu == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#opens a json file where the category names is located.
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

#loads the model from the checkpoint
model = load_checkpoint(args.checkpoint)
model.to(device)    

image_path = args.image_path
names, classes, props = predict(image_path, model, cat_to_name, args.topk)
props = props.squeeze().tolist()        #converting from tensor to list

print("the following categories represent the predictions with the {} highest probabilities in descending order:".format(args.topk))
for i in range(args.topk):
    print("Category: {} .. Probability: {:.2f}%".format(names[i],  props[i]*100))