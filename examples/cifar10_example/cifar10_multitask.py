# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # CIFAR-10 Multi-Task
#
# In this example we're going to build the common exercise of building a CIFAR-10 classifier but with a multi-task twist: in addition to predicting the class, we're also going to estimate the average RGB values of each image.
# This will turn our classic classification example into a classification and regression multi-task learning problem.

# + [markdown] tags=["md-exclude"]
# ## Environment Setup
# -

# %load_ext autoreload
# %autoreload 2

# + tags=["md-exclude"]
import os
import sys

# NOTE: for dev purposes add package to path
cerbero_path = os.path.normpath(os.path.join(os.getcwd(), "../../"))
sys.path.append(cerbero_path)

import numpy as np
import random
import torch
import matplotlib.pyplot as plt
# %matplotlib inline
# -

# ## Create Dataset

# The CIFAR-10 dataset is a set of 3-channel color images of 32x32 pixels in size.
# Each image is labeled with one of the following classes:  ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.
#
# In addition to predicting the correct label for each image we're also going to estimate the average RGB color for each image. For this we'll have to do a bit of additional preprocessing to get to the following dataset:
#
# - input: [3,32,32] float tensor of normalized range [-1,1]
# - targets:
#   - class label: int [0,9]
#   - rgb: 3-element vector of range [0,1]

import torchvision
import torchvision.transforms as transforms

# +
from utils import make_cifar_dataset
from sklearn.model_selection import train_test_split

images, class_labels, rgb_labels = make_cifar_dataset(train=True)

# Further split train into train/test
(images_train,
 images_test,
 class_labels_train,
 class_labels_test,
 rgb_labels_train,
 rgb_labels_test) = train_test_split(images, class_labels, rgb_labels, test_size=0.1)

images_val, class_labels_val, rgb_labels_val = make_cifar_dataset(train=False)
# -

# We can now group our inputs and labels based on the tasks.
# Normally a dictionary would work just fine, but in our case we need to use the same input data but with multiple output labels.
# To avoid making a copy of our data and reducing our memory footprint it's recomended to use the `cerbero.core.Database`.

# The `cerbero.core.Database` object is a simple relation database that allows us to map multiple tasks to the same input data. 

# Warning: `cerbero.core.Database` expects all values to be `torch.Tensor` objects. 

# +
# Convert datasets to torch.Tensors
images_train = torch.Tensor(images_train)
class_labels_train = torch.Tensor(class_labels_train).type(torch.LongTensor)
rgb_labels_train = torch.Tensor(rgb_labels_train)

images_test = torch.Tensor(images_test)
class_labels_test = torch.Tensor(class_labels_test).type(torch.LongTensor)
rgb_labels_test = torch.Tensor(rgb_labels_test)

images_val = torch.Tensor(images_val)
class_labels_val = torch.Tensor(class_labels_val).type(torch.LongTensor)
rgb_labels_val = torch.Tensor(rgb_labels_val)

# +
from cerbero.core import Database

X_train, X_val, X_test = Database(), Database(), Database()
Y_train, Y_val, Y_test = Database(), Database(), Database()

# Label class dataset
X_train["class"] = images_train
Y_train["class"] = class_labels_train
X_test["class"] = images_test
Y_test["class"] = class_labels_test
X_val["class"] = images_val
Y_val["class"] = class_labels_val

# RGB dataset
X_train["rgb"] = images_train
Y_train["rgb"] = rgb_labels_train
X_test["rgb"] = images_test
Y_test["rgb"] = rgb_labels_test
X_val["rgb"] = images_val
Y_val["rgb"] = rgb_labels_val
# -

# ## Make DataLoaders

# With our data now loaded/created, we can now package it up into `DictDataset` objects for training.
# This object is a simple wrapper around `torch.utils.data.Dataset` and stored data fields and labels as dictionaries.
#
# In the `DictDataset`, each label corresponds to a particular `Task` by name. We'll define these `Task` objects in the following section as we define our model.
#
# `DictDataLoader` is a wrapper for `torch.utils.data.DataLoader`, which handles the collate function for `DictDataset` appropriately.

# +
from cerbero.core import DictDataset, DictDataLoader

dataloaders = []
for task_name in ["class", "rgb"]:
    for split, X, Y in (
        ("train", X_train, Y_train),
        ("valid", X_val, Y_val),
        ("test", X_test, Y_test)
    ):
        X_dict = {f"{task_name}_data": torch.FloatTensor(X[task_name])}
        YTensor = torch.FloatTensor if task_name == "rgb" else torch.LongTensor
        Y_dict = {f"{task_name}_task": YTensor(Y[task_name])}
        dataset = DictDataset(f"{task_name}Dataset", split, X_dict, Y_dict)
        dataloader = DictDataLoader(dataset, batch_size=32)
        dataloaders.append(dataloader)
# -

# We now have 4 data loaders, one for each split (`train`, `val`) of each task (`class_task` and `rgb_task`)

# ## Define Model

# Now we'll define the `MultitaskClassifier` model, a PyTorch multi-task classifier. We'll instantiate it from a list of `Tasks`

# +
import torch.nn as nn
from cerbero.core import Operation

# Define the base conv net and a one-layer prediction "head" module
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

base_net = BaseNet()
class_head_module = nn.Linear(84, 10)

# The module pool contains all the modules this task uses
module_pool = nn.ModuleDict(
    {"base_net": base_net, "class_head_module": class_head_module}
)

# Op1: pull data from "class_data" and send through base net
op1 = Operation(
    name="base_net",
    module_name="base_net",
    inputs=[("_input_", "class_data")]
)

# Op2: pass output of Op1 to the class head module
op2 = Operation(
    name="class_head",
    module_name="class_head_module",
    inputs=["base_net"]
)

op_sequence = [op1, op2]
# -

# The output of the final operation will then go into a loss function to calculate the loss (e.g., cross-entropy) during training or an output function (e.g., softmax) to convert the logits into a prediction.
#
# Each `Task` also specifies which metrics it supports, which are bundled together in a `Scorer` object. For this tutorial, we'll just look at accuracy.

# +
from functools import partial

import torch.nn.functional as F

from cerbero.metrics import Scorer
from cerbero.core import Task

class_task = Task(
    name="class_task",
    module_pool=module_pool,
    op_sequence=op_sequence,
    loss_func=F.cross_entropy,
    output_func=partial(F.softmax, dim=1),
    scorer=Scorer(metrics=["accuracy"]),
)


# -

# ### Again, for the RGB `Task`

# In this case, the RGB `Task` differs in that we'll be training the model to estimate the average RGB colors in the image which we model here as a regression task. Additonally, we'll define the RGB head as a two-layer module.

# +
class RGBHead(nn.Module):
    def __init__(self):
        super(RGBHead, self).__init__()
        self.fc1 = nn.Linear(84, 16)
        self.fc2 = nn.Linear(16, 3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        return x

rgb_head_module = RGBHead()

op_sequence = [
    Operation(
        name="base_net",
        module_name="base_net",
        inputs=[("_input_", "rgb_data")]
    ),
    Operation(
        name="rgb_head",
        module_name="rgb_head_module",
        inputs=["base_net"]
    )
]

def identity_fn(x):
    return x

rgb_task = Task(
    name="rgb_task",
    module_pool=nn.ModuleDict(
        {"base_net": base_net,
         "rgb_head_module": rgb_head_module}
    ),
    op_sequence=op_sequence,
    loss_func=F.mse_loss,
    output_func=identity_fn,
    scorer=Scorer(metrics=["mse"])
)
# -

# ## Model

# With our tasks defined, constructing a model is simple: we simply pass the list of tasks in and the model constructs itself using information from the task flows.
#
# Note that the model uses the names of modules (not the modules themselves) to determine whether two modules specified by separate tasks are the same module (and should share weights) or different modules (with separate weights).
# So because both the `class_task` and `rgb_task` include "base_net" in their module pools, this module will be shared between the two tasks.

# +
from cerbero.models import MultitaskModel

model = MultitaskModel([class_task, rgb_task])
# -

# ### Train Model

# Once the model is constructed, we can train it as we would a single-task model, using the `fit` method of a `Trainer` object. The `Trainer` supports multiple schedules or patterns for sampling from different dataloaders; the default is to randomly sample from them proportional to the number of batches, such that all data points will be seen exactly once before any are seen twice.

# +
from cerbero.trainer import Trainer

trainer_config = {
    "progress_bar": True,
    "n_epochs": 15,
    "lr": 2e-3,
    "checkpointing": True
}

trainer = Trainer(**trainer_config)
trainer.fit(model, dataloaders)
# -

# ### Evaluate the model

# After training, we can call the `model.score()` method to see the final performance on all tasks

model.score(dataloaders)

# ### Inspect model predictions

dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)

# +
images = dataset.data[:16]
rgb_labels = images.mean(axis=(1,2))
class_labels = dataset.targets[:16]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(dataloader)
torch_images, _ = next(dataiter)
norm_rgb_labels = torch_images.mean(dim=(2,3))

norm_transform = transforms.Normalize(
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
)
norm_images = torch.stack(
    [
        norm_transform(torch.Tensor(_image))
        for _image in torch_images
    ],
    dim=0
)
# -

input_dict = {
    "class_data": norm_images,
    "rgb_data": norm_images
}
with torch.no_grad():
    out_dict = model(input_dict, task_names=["class_task", "rgb_task"])

class_out = torch.argmax(
    F.softmax(out_dict["class_head"], dim=1),
    dim=1
)
rgb_out = out_dict["rgb_head"]

# +
print(f"class_head")
print(f"y_true: {class_labels}")
print(f"y_pred: {class_out.numpy().tolist()}")

print(f"\nrgb_head")
print(f"y_true:\n{np.round(rgb_labels).astype(int)}")
print(f"y_pred:\n{np.round(rgb_out.numpy()*255).astype(int)}")
