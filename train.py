import os
import sys
import time
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from tqdm import tqdm

import argparse

training_loader = 
testing_loader = 

batch_size = 64
max_lr = .05
n_epochs = 100
n_runs = 1

steps_per_epoch = len(training_loader)

for i_run in range(n_runs):
    network = 
    # print(f"Model has {network.n_params()} parameters
    optimizer = optim.Adam(network.parameters(), lr=max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs)

    epoch_accuracy_history = []
    for i_epoch in range(n_epochs):

        epoch_start_time = time.time()
        epoch_training_loss = 0
        epoch_testing_loss = 0
        epoch_training_num_correct = 0
        epoch_testing_num_correct = 0

        with tqdm(enumerate(training_loader)) as tqdm_training_loader:
            for batch_idx, batch in tqdm_training_loader:

                images, labels = batch
                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_training_loss += loss.item() * training_loader.batch_size
                epoch_training_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

                tqdm_training_loader.set_description(
                    f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                    f'Epoch: {i_epoch + 1}/{n_epochs}, '
                    f'Run: {i_run + 1}/{n_runs}'
                )

        epoch_duration = time.time() - epoch_start_time
        training_loss = epoch_training_loss / len(training_loader.dataset)
        training_accuracy = (epoch_training_num_correct / len(training_loader.dataset))

        # At the end of each epoch run the testing data through an
        # evaluation pass to see how the model is doing.
        # Specify no_grad() to prevent a nasty out-of-memory condition.
        with torch.no_grad():
            for batch in testing_loader:
                images, labels = batch
                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                epoch_testing_loss += loss.item() * testing_loader.batch_size
                epoch_testing_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

            testing_loss = epoch_testing_loss / len(testing_loader.dataset)
            testing_accuracy = (epoch_testing_num_correct / len(testing_loader.dataset))
            epoch_accuracy_history.append(testing_accuracy)

        print(
            f"run: {i_run}   "
            f"epoch: {i_epoch}   "
            f"duration: {epoch_duration:.04}   "
            f"learning rate: {scheduler.get_last_lr()[0]:.04}   "
            f"training loss: {training_loss:.04}   "
            f"testing loss: {testing_loss:.04}   "
            f"training accuracy: {100 * training_accuracy:.04}%   "
            f"testing accuracy: {100 * testing_accuracy:.04}%"
        )



