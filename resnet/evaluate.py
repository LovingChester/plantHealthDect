import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Dict, List
from resnet18 import ResNet, BasicBlock
from utils import get_data
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassRecall, MulticlassPrecision
from mlxtend.plotting import plot_confusion_matrix
plt.style.use('ggplot')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def visualize_prediction(model: torch.nn.Module, 
                         image_path: str = None, 
                         class_names: List[str] = None,
                         T: int = 1000):
    '''
    Makes a prediction on a target image and plots the image with its prediction.
    '''

    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model's expected input size
        transforms.ToTensor(),           # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        output = model(input_batch.to(device))
    pred_probs = torch.softmax(output, dim=1)
    pred_label_idx = torch.argmax(pred_probs).item()
    pred_label = class_names[pred_label_idx]
    # model = model.train()

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    mc_dropout_results = np.zeros((T, len(class_names)))
    for i in range(T): 
        with torch.inference_mode():
            output = model(input_batch.to(device))
        mc_dropout_results[i] = torch.softmax(output, dim=1).cpu().numpy()
    model.train()

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Pred: {pred_label} \n Actual: {image_path.rsplit("/", 2)[-2]}')
    plt.subplot(1, 2, 2)
    plt.boxplot(mc_dropout_results, vert=False)
    plt.yticks(range(1, len(class_names) + 1), class_names)
    plt.subplots_adjust(wspace=1.5)
    plt.savefig(os.path.join('outputs', 'pred.png'))


def metrics(model: torch.nn.Module,
                     test_data,
                     class_names: List[str] = None):
    model.eval()
    preds = []
    targets = []
    test_running_correct = 0
    with torch.inference_mode():
        for X, y in tqdm(test_data, desc="Making predictions"):
            X, y = X.to(device), y.to(device)
            output = model(X)
            pred = torch.softmax(output, dim=1).argmax(dim=1)
            preds.append(pred.cpu())
            targets.append(y.cpu())
            test_running_correct += (pred == y).sum().item()
    acc = (test_running_correct / len(test_data.dataset))
    preds_tensor = torch.cat(preds)
    targets_tensor = torch.cat(targets)

    print(f'Accuracy: {acc}')
    metric = MulticlassPrecision(num_classes=len(class_names), average='weighted')
    print(f'Precision: {metric(preds_tensor, targets_tensor).cpu().numpy()}')
    metric = MulticlassRecall(num_classes=len(class_names), average='weighted')
    print(f'Recall: {metric(preds_tensor, targets_tensor).cpu().numpy()}')
    metric = MulticlassMatthewsCorrCoef(num_classes=len(class_names))
    print(f'Matthew’s Correlation Coefficient: {metric(preds_tensor, targets_tensor).cpu().numpy()}')
    metric = MulticlassF1Score(num_classes=len(class_names), average='weighted')
    print(f'F1 Score: {metric(preds_tensor, targets_tensor).cpu().numpy()}')
    
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=preds_tensor,
                             target=targets_tensor)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(30, 30)
    )
    plt.savefig(os.path.join('outputs', 'cm.png'))


if __name__ == '__main__':
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=39)
    model.load_state_dict(torch.load(f='outputs/resnet18.pth'))
    model.to(device)

    folder_path = './data/test'
    classes = sorted([f.name for f in os.scandir(folder_path)])
    # random_class = random.choice(classes)
    # # random_class = classes[1]
    # imgs = [f for f in os.listdir(os.path.join(folder_path, random_class))]
    # random_img = os.path.join(folder_path, random_class, random.choice(imgs))

    # visualize_prediction(model=model,
    #                      image_path=random_img,
    #                      class_names=classes)

    test_loader = get_data(background=False, batch_size=64)['valid']
    metrics(model=model,
            test_data=test_loader,
            class_names=classes)