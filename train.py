from tqdm import tqd
from datetime import datetime
import argparse
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from data_module import FrameDataset
from ff_network import Net

from train_utils import createdirs, log



parser = argparse.ArgumentParser(description='SCS Train')
parser.add_argument('--model', default='', choices = ['rnn', 'ff'], help='model type')
parser.add_argument('--dataset', default='', help='model type')
args = parser.parse_args()

#check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#get dataset
dataset = args.dataset

#read in data and create dataloaders
x_train = pd.read_csv(f'Datasets/{dataset}_x_train.csv')
y_train = pd.read_csv(f'Datasets/{dataset}_y_train.csv')
x_test = pd.read_csv(f'Datasets/{dataset}_x_test.csv')
y_test = pd.read_csv(f'Datasets/{dataset}_y_test.csv')

train = FrameDataset(x_train,y_train)
test = FrameDataset(x_test,y_test)

training_loader = DataLoader(train)
testing_loader = DataLoader(test)

#model params
batch_size = 2048
max_lr = .05
n_epochs = 100
n_runs = 1
input_size = x_train.shape[1]
output_size = y_train.shape[1]
h_size = 64

network_gen = {
    "ff": Net(input_size=input_size, output_size=output_size,hidden_size=h_size)
}

model_gen = network_gen.get(args.model)

steps_per_epoch = len(training_loader)

network = model_gen().to(device)
print(f"Training: {args.model}")

#maybe add option for SGD instead
optimizer = optim.Adam(network.parameters(), lr=max_lr)

path = 'logs/' + args.model + '/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
createdirs(path)
log_file = open(path + '.log', 'w+')

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

            frames, labels = batch
            preds = network(frames)
            loss = F.mse_loss(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_training_loss += loss.item() * training_loader.batch_size
            epoch_training_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

            tqdm_training_loader.set_description(
                f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                f'Epoch: {i_epoch + 1}/{n_epochs}, '
            )

    epoch_duration = time.time() - epoch_start_time
    training_loss = epoch_training_loss / len(training_loader.dataset)
    training_accuracy = (epoch_training_num_correct / len(training_loader.dataset))

    #test set
    #maybe change this to do less often for speed
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

    log('Epoch {} Train Loss {} Train Accuracy {} Test Loss {} Test Accuracy {}'.format(
    i_epoch, training_loss, training_accuracy, testing_loss, testing_accuracy), log_file)

#save model checkpoint
torch.save(network.state_dict(), path + '.pt')
log_file.close()