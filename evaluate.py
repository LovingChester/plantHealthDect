import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vit_b_16
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from data_loaders import get_data
from config import cfg

def evaluate():

    # define model
    if cfg.model.name == 'vgg16':
        model = vgg16(weights='DEFAULT')
        model.classifier[6] = nn.Linear(4096, 39, bias=True)
    elif cfg.model.name == 'vit':
        model = vit_b_16(weights='DEFAULT')
        model.heads.head = nn.Linear(768, 39, bias=True)

    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"using device: {device}")
    model.to(device)

    out_dir = cfg.train.out_dir
    if cfg.eval.mode == 'best':
        fname = os.path.join(out_dir, 'model_dict.pth')
    elif cfg.eval.mode == 'final':
        fname = os.path.join(out_dir, 'model_dict_final.pth')
    model.load_state_dict(torch.load(fname))
    model.eval()

    _, val_loader, test_loader = get_data(cfg)

    # keep track of prediction and true label
    val_y_pred = []
    val_y_true = []

    with torch.no_grad():
        for i, (image_data, label) in enumerate(val_loader):
            
            image_data = image_data.to(device)
            #label = label.to(device)

            output = model(image_data)
            pred = F.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().detach().numpy().tolist()
            val_y_pred.extend(pred)
            val_y_true.extend(label)
    
    val_acc = accuracy_score(val_y_true, val_y_pred)
    print(val_acc)

    test_y_pred = []
    test_y_true = []

    with torch.no_grad():
        for i, (image_data, label) in enumerate(test_loader):
            
            image_data = image_data.to(device)
            #label = label.to(device)

            output = model(image_data)
            pred = F.softmax(output, dim=1)
            pred = torch.argmax(pred, dim=1)

            pred = pred.cpu().detach().numpy().tolist()
            test_y_pred.extend(pred)
            test_y_true.extend(label)
    
    test_y_pred = np.array(test_y_pred)
    test_y_true = np.array(test_y_true)
    test_acc = accuracy_score(test_y_true, test_y_pred)
    print(test_acc)

    cm = confusion_matrix(test_y_true, test_y_pred)

    # stores the accuracy for each class
    acc_class = []
    for i in range(39):
        acc = cm[i, i] / sum(cm[i])
        acc_class.append(acc)
        print('class {}: {:.3f}'.format(i, acc))

    print('average accuracy of class accuracy: {:.3f}'.format(np.mean(acc_class)))

if __name__ == "__main__":
    evaluate()
