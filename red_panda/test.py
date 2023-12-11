import itertools

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


def show_accuracy_matrix(model, dataloader, device, class_names):
    # Переводим модель в режим оценки
    model.eval()
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)  # Feed Network
        output = (torch.max(output, 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')


if __name__ == '__main__':
    model = models.efficientnet_b2()
    num_ftrs = model.classifier[1].in_features
    model.fc = torch.nn.Linear(num_ftrs, 11)
    model.load_state_dict(torch.load(r'D:\matvey\dataset\best_model_params.pt'))
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'D:\matvey\dataset\dataset_n'
    image_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms["test"])
    dataloaders = torch.utils.data.DataLoader(image_dataset, batch_size=4,
                                              shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Перемещаем модель на устройство
    model = model.to(device)
    class_names = image_dataset.classes
    print(class_names)
    show_accuracy_matrix(model, dataloaders,device,class_names)
