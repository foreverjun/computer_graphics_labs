import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model_predictions(model, img_path):
    was_training = model.training
    model.eval()
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    img = Image.open(img_path)
    img = data_transforms['test'](img)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'rpanda',
                   'scoiattolo']
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2, 2, 1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)


if __name__ == '__main__':
    model = models.efficientnet_b2()
    num_ftrs = model.classifier[1].in_features
    model.fc = torch.nn.Linear(num_ftrs, 11)
    model.load_state_dict(torch.load(r'D:\matvey\dataset\best_model_params.pt'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    visualize_model_predictions(
        model,
        img_path=r'C:\Users\kalas\Downloads\images (1).jpg'
    )
    visualize_model_predictions(
        model,
        img_path=r'C:\Users\kalas\Downloads\images.jpg'
    )

    plt.ioff()
    plt.show()
