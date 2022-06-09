import torch
import argparse
import pytorch_lightning as pl
import time
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint

from models import MarsModel
start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--outputs-dir', type=str, required=True)
parser.add_argument('--num-epochs', type=int, required=True)
args = parser.parse_args()


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

checkpoint = ModelCheckpoint(verbose=True, monitor="val_acc", mode="max")

data_transform = transforms.Compose(
    [
        transforms.Resize([220, 220]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
roottrain=os.path.join(args.data_dir,"train")
roottest=os.path.join(args.data_dir,"test")
rootval=os.path.join(args.data_dir,"val")
ctx_train = datasets.ImageFolder(root=roottrain, transform=data_transform)
train_loader = torch.utils.data.DataLoader(
    ctx_train,
    batch_size=hyper_params["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

ctx_val = datasets.ImageFolder(root=rootval, transform=data_transform)
val_loader = torch.utils.data.DataLoader(
    ctx_val, batch_size=hyper_params["batch_size"], shuffle=True, num_workers=8
)

ctx_test = datasets.ImageFolder(root=roottest, transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    ctx_test, batch_size=16, shuffle=True, num_workers=4
)
print(roottrain,roottest,rootval)

model = MarsModel(hyper_params)

trainer = pl.Trainer(
    gpus=1, max_epochs=hyper_params["num_epochs"], checkpoint_callback=checkpoint
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(dataloaders=test_loader)
