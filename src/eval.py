import torch
import argparse
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from torchvision import datasets, transforms
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix

from models import MarsModel
from utils import onehot


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--weights-dir', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--num-epochs', type=int, required=True)
parser.add_argument('--img-size', type=int, required=True)
args = parser.parse_args()
network_name = args.model

data_transform = transforms.Compose(
    [
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

hyper_params = {
    "batch_size": 64,
    "num_epochs": args.num_epochs,
    "learning_rate": 1e-2,
    "optimizer": "sgd",
    "momentum": 0.9,
    "model": args.model,
    "num_classes": 15,
    "pretrained": False,
    "transfer_learning": False,
}
roottest=os.path.join(args.data_dir,"test")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MarsModel(hyper_params)
model.load_state_dict(torch.load(args.weights_dir))
ctx_test = datasets.ImageFolder(root=roottest, transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    ctx_test, batch_size=16, shuffle=True, num_workers=4
)

labels = []
predictions = []
scores = []

# Put on GPU if available
model = model.to(device)

# Set model to eval mode (turns off dropout and moving averages of batchnorm)
model.eval()

# Iterate over test set
with torch.no_grad():
    for i_batch, batch in enumerate(test_loader):
        x, y = batch
        y_hat = model(x.to(device))
        pred = torch.argmax(y_hat, dim=1).cpu()

        labels.append(y.numpy())
        predictions.append(pred.numpy())
        scores.append(F.softmax(y_hat, dim=1).detach().cpu().numpy())

# Computing metrics
labels = np.concatenate(labels, axis=0)
predictions = np.concatenate(predictions, axis=0)
scores = np.concatenate(scores, axis=0)

onehot_labels = [onehot(label, hyper_params["num_classes"]) for label in labels]
onehot_predictions = [
    onehot(prediction, hyper_params["num_classes"]) for prediction in predictions
]

y = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
y_pred = label_binarize(
    predictions, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
)

macro_roc_auc_ovo = roc_auc_score(y, scores, multi_class="ovo", average="macro")

macro_roc_auc_ovr = roc_auc_score(y, scores, multi_class="ovr", average="macro")

acc = metrics.accuracy_score(y, y_pred)

# Writing results to file
print(
    "Classification report for classifier %s:\n%s\n"
    % (network_name, metrics.classification_report(y, y_pred, digits=4)),
    file=open(args.outputs_dir+'/' + network_name + ".txt", "w"),
)
print(
    "AUROC:\t",
    macro_roc_auc_ovo,
    macro_roc_auc_ovr,
    file=open(args.outputs_dir+'/' + network_name + ".txt", "a"),
)
print("Acc:\t", acc, file=open("./results/" + network_name + ".txt", "a"))
print("\n", file=open(args.outputs_dir+'/' + network_name + ".txt", "a"))
print(
    confusion_matrix(labels, predictions),
    file=open(args.outputs_dir+'/' + network_name + ".txt", "a"),
)
