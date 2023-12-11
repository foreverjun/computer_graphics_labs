import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import torch
from PIL import Image
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
from torchvision.models.efficientnet import EfficientNet_B2_Weights

cudnn.benchmark = True


def resize_image(input_path, output_path, size):
    with Image.open(input_path) as image:
        image = image.resize((300, 300))
        image.save(output_path)


# Функция для рекурсивного обхода папок
def process_directory(input_dir, output_dir, size):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if os.path.isdir(input_path):
            os.makedirs(output_path, exist_ok=True)
            process_directory(input_path, output_path, size)
        else:
            resize_image(input_path, output_path, size)


def recursive():
    train = r"D:\matvey\dataset\dataset_n\train"
    test = r"D:\matvey\dataset\dataset_n\test"
    valid = r"D:\matvey\dataset\dataset_n\valid"
    old = r"D:\matvey\dataset\raw-img_n"
    for directory in os.listdir(old):
        if os.path.isdir(os.path.join(old, directory)):
            files = os.listdir(os.path.join(old, directory))
            train_data, valid_data = train_test_split(files, test_size=0.3, random_state=42)
            os.makedirs(os.path.join(train, directory), exist_ok=True)

            for i in train_data:
                shutil.copy(os.path.join(old, directory, i), os.path.join(train, directory, i))
            os.makedirs(os.path.join(valid, directory), exist_ok=True)
            for i in valid_data:
                shutil.copy(os.path.join(old, directory, i), os.path.join(valid, directory, i))
            valid_data, test_data = train_test_split(os.listdir(os.path.join(valid, directory)), test_size=0.3,
                                                     random_state=42)
            os.makedirs(os.path.join(test, directory), exist_ok=True)
            for i in test_data:
                shutil.move(os.path.join(valid, directory, i), os.path.join(test, directory, i))


def split_dataset():
    train = r"D:\matvey\dataset\dataset_n\train"
    test = r"D:\matvey\dataset\dataset_n\test"
    valid = r"D:\matvey\dataset\dataset_n\valid"
    os.makedirs(r"D:\matvey\dataset\dataset_n\train", exist_ok=True)
    os.makedirs(r"D:\matvey\dataset\dataset_n\test", exist_ok=True)
    os.makedirs(r"D:\matvey\dataset\dataset_n\valid", exist_ok=True)

    recursive()


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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        print(tempdir)
        best_model_params_path = os.path.join("D:\matvey\dataset", 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

if __name__ == '__main__':
    # width = 300
    # height = 300
    # input_shape = (height, width, 3)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'D:\matvey\dataset\dataset_n'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


    model_ft = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    num_ftrs = model_ft.classifier[1].in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=10)

    visualize_model(model_ft)