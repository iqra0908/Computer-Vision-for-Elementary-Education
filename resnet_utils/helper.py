## Library imports
from torchvision import models, transforms
import torch
from PIL import Image
from config import classes_imagenet

## Load the model class names
with open(classes_imagenet) as f:
    classes = [line.strip() for line in f.readlines()]

## Load the model
def load_resnet18():
    """
    The main function to load the resnet18 model
    """
    resnet = models.resnet18(pretrained=True)
    resnet.eval()

    return resnet

def predict(image_path):
    """
    The main function to predict the class of the image using a resnet18 model
    """
    ## Load the model
    resnet = load_resnet18()

    ## Define transforms to be applied on the image
    transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                ])

    ## Load the image and apply the transforms
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    
    ## Pass the image through the model
    out = resnet(batch_t)

    ## Get the probabilities of the classes
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]