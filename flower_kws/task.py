"""flower-kws: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
from datasets import concatenate_datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from torch.utils.data import DataLoader

from flower_kws.audio import (
    balance_dataset,
    flatten_all_segments,
    get_apply_transforms_fn,
    segment_silences,
)

REMOVE_COLS = ["file", "audio", "label", "is_unknown", "speaker_id", "utterance_id"]


class Net(nn.Module):
    def __init__(self, numClasses: int = 12):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
        )
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        self.fc = nn.Linear(128 * 2 * 3, numClasses)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.pool(self.relu(self.layer2(x)))
        x = self.relu(self.layer3(x))
        x = self.fc(torch.flatten(x, 1))
        return x


fds = None  # Cache FederatedDataset


def load_data(partition_id: int):
    """Load partition SpeechCommands data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="speaker_id", group_size=20
        )
        fds = FederatedDataset(
            dataset="speech_commands",
            subset="v0.02",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )
    partition = fds.load_partition(partition_id)

    pp = partition.map(
        get_apply_transforms_fn(), batch_size=32, remove_columns=REMOVE_COLS
    )

    # Now let's add some _silence_ training examples
    silences = fds.partitioners["train"].dataset.filter(lambda x: x["label"] == 35)
    silence_segments = silences.map(segment_silences(int(len(pp) * 0.02)))
    flattened_silences_dataset = silence_segments.map(
        flatten_all_segments, batched=True
    )
    silence_enc = flattened_silences_dataset.map(
        get_apply_transforms_fn(), batch_size=32, remove_columns=REMOVE_COLS
    )
    # Concatenate dataset
    pp = concatenate_datasets([pp, silence_enc])

    # Prep for Torch
    pp.set_format(type="torch", columns=["mfcc", "target"])

    # Ensure dataset partition is balanced
    balanced_dataset = balance_dataset(pp, target_column="target")
    pp_train_val = balanced_dataset.train_test_split(test_size=0.1)
    trainloader = DataLoader(pp_train_val["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(pp_train_val["test"], batch_size=32)

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["mfcc"]
            labels = batch["target"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["mfcc"].to(device)
            labels = batch["target"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_device():
    return (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
