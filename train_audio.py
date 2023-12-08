#!/usr/bin/env python3


import math
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
from sklearn.metrics import roc_auc_score

from scipy import signal
import torch
import pickle
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import MagnaTagATune
from evaluation import evaluate

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a simple CNN on audio",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


default_dataset_dir = Path.home() / ".cache" / "torch" / "datasets"
parser.add_argument("--dataset-root", default=default_dataset_dir)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.05, type=float, help="Learning rate")
parser.add_argument("--momentum", default=0.94, type=float, help="momentum")

parser.add_argument("--extension", default=False, type=bool, help="extension")

parser.add_argument("--length", default=256, type=int, help="length")
parser.add_argument("--stride", default=256, type=int, help="stride")



parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of images within each mini-batch",
)

parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)


class GlobalMinMax:
    def __init__(self):
        self.min = float('inf')
        self.max = float('-inf')

global_min_max = GlobalMinMax()


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    transform = transforms.ToTensor()
    args.dataset_root.mkdir(parents=True, exist_ok=True)


    train_dataset= MagnaTagATune(dataset_path='/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/train_labels.pkl', samples_path='/mnt/storage/scratch/gv20319/MagnaTagATune/samples/train')
    compute_min_max(train_dataset)

    
    train_dataset= MagnaTagATune(dataset_path='/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/train_labels.pkl', samples_path='/mnt/storage/scratch/gv20319/MagnaTagATune/samples/train', global_min= global_min_max.min, global_max= global_min_max.max)

    validation_dataset = MagnaTagATune(dataset_path='/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/validation.pkl', samples_path='/mnt/storage/scratch/gv20319/MagnaTagATune/samples/val' ,global_min= global_min_max.min, global_max= global_min_max.max)

    test_dataset = MagnaTagATune(dataset_path='/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/val_labels.pkl', samples_path='/mnt/storage/scratch/gv20319/MagnaTagATune/samples/valtest', global_min= global_min_max.min, global_max= global_min_max.max)

   
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    specmodel = SpecCNN(length = args.length , channels=1, class_count=50, stride=args.stride)
    basemodel = CNN(length = args.length , channels=1, class_count=50, stride=args.stride)

    if args.extension == True:
        model = specmodel
    else:
        model = basemodel
    
    criterion = nn.BCELoss()

    
    optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, validation_loader, test_loader, criterion, optimizer, scheduler, summary_writer, DEVICE
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    summary_writer.close()





class SpecCNN(nn.Module):
    def __init__(self, length: int, channels: int, class_count: int, stride: int):
        super().__init__()

        #conv1
        self.conv1 = nn.Conv1d( in_channels=1, out_channels=32, kernel_size=8, padding= "same" )
        self.bn1 = nn.BatchNorm1d(32)
        self.initialise_layer(self.conv1)
     
        #pool1
        self.pool1 = nn.MaxPool1d(kernel_size=4 )
        
        ##conv2
        self.conv2 = nn.Conv1d( in_channels=32, out_channels=32, kernel_size=8, padding= "same" )
        self.bn2 = nn.BatchNorm1d(32)
        self.initialise_layer(self.conv2)

        #pool2
        self.pool2 = nn.MaxPool1d(kernel_size=4)
      
        #fc1
        #35072
        self.fc1 = nn.Linear(35072, 100)
        self.initialise_layer(self.fc1)

        self.bn3 = nn.BatchNorm1d(100)

        self.dropout = nn.Dropout(p=0.5)

        #fc2
        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

        

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        #Move the tensor to CPU and convert it to numpy
        input_cpu = input.cpu().numpy()

        f, t, Sxx = signal.spectrogram(input_cpu, fs=12000, nperseg=256, noverlap=0)
        Sxx = np.log1p(10000 * Sxx)  # Dynamic range compression

        x = torch.from_numpy(Sxx)

        x = x.to(input.device)

        batch_size = input.size(0)

        x  = torch.flatten(x, start_dim=0, end_dim=1)

        # Flatten the last two dimensions
        x = torch.flatten(x, start_dim= -2)

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = self.pool2(x)
      
        x = torch.flatten(x, start_dim=1) 

        x= F.relu(self.fc1(x))
        x = self.bn3(x)
        
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
      
        x = torch.reshape(x, (batch_size, -1, 50))
        x = x.mean(dim=1)

        return x


    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)



class CNN(nn.Module):
    def __init__(self, length: int, channels: int, class_count: int, stride: int):
        super().__init__()
        
        #strided convolution
        self.conv0 = nn.Conv1d(

            in_channels= channels,
            out_channels=32,
            kernel_size=length,
            stride= stride,
            
        )
        #self.bn1 = nn.BatchNorm1d(32)
        self.initialise_layer(self.conv0)


        #REST OF THE LAYERS

        #conv1
        self.conv1 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=8,
            padding= "same",
        )
        #self.bn2 = nn.BatchNorm1d(32)
        self.initialise_layer(self.conv1)
     
        #pool1
        self.pool1 = nn.MaxPool1d(kernel_size=4 )
        
        ##conv2
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=8,
            padding= "same",
        )
        self.initialise_layer(self.conv2)

        #pool2
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        inputtofc1 = self.convlayeroutput(34950, length, stride, 4)

        #fc1 35072
        self.fc1 = nn.Linear(35072, 100)
        self.initialise_layer(self.fc1)

        #fc2
        self.fc2 = nn.Linear(100, 50)
        self.initialise_layer(self.fc2)

    #function to calculate the output of all the conv layers
    def convlayeroutput(self, initial_size, conv_kernel_size, stride, pool_kernel_size):

        current_size = initial_size

        current_size = math.floor((current_size - conv_kernel_size ) / stride + 1)

        if current_size <= 0:
            raise ValueError("The size of the conv layer is less than 0")
        
        for _ in range(2):
            
            current_size = math.floor((current_size - pool_kernel_size ) / pool_kernel_size + 1)
            if current_size <= 0:
                raise ValueError("The size after pooling is less than 0")
        
        return current_size*32


    def forward(self, input: torch.Tensor) -> torch.Tensor:


        batch_size = input.size(0)

        x  = torch.flatten(input, start_dim=0, end_dim=1)


        x = F.relu(self.conv0(x))

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
      
        x = torch.flatten(x, start_dim=1) 

        x= F.relu(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))
      
        x = torch.reshape(x, (batch_size, -1, 50))
        x = x.mean(dim=1)

        return x


    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            all_preds = []
            data_load_start_time = time.time()

            
           
            for filename, batch, labels  in self.train_loader:

                batch = batch.to(self.device)

                labels = labels.to(self.device)
                data_load_end_time = time.time()

                logits = self.model.forward(batch)

                loss = self.criterion(logits, labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                
                with torch.no_grad():
                    #preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, logits)
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()
                
            #scheduler step
            self.scheduler.step()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate(self.val_loader, '/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/validation.pkl')
                #self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

        self.validate(self.test_loader, '/mnt/storage/scratch/gv20319/MagnaTagATune/annotations/val_labels.pkl')
        
        
    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self, data_loader, file_path):
        results = {"logits": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for filename, batch, labels in data_loader:
                
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                results["logits"].extend(list(logits.cpu().numpy()))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["logits"])
        )

        average_loss = total_loss / len(data_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        # Call evaluate function
        evaluate(results["logits"], file_path)
      
        print('average loss: {:.4f}'.format(average_loss))


def compute_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray])-> float:
    """
    Compute the average AUC-ROC score across all labels.
    
    Args:
        labels: (batch_size, class_count) tensor containing true labels.
        preds: (batch_size, class_count) tensor containing model predictions.

    Returns:
        avg_auc_score: Average AUC score across all labels.

    """
    # Check if inputs are tensors, convert them to array
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    

    # Calculate AUC for each label and handle exceptions
    auc_scores = []
    for i in range(labels.shape[1]):  # Iterate over each label
        try:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            auc_scores.append(auc)
        except ValueError:
            pass  # Handle labels with no positive examples

    # Calculate average AUC score
    if auc_scores:  # Check if list is not empty
        avg_auc_score = np.mean(auc_scores)
    else:
        avg_auc_score = 0.0  # Default score if no valid AUC scores were found

    return avg_auc_score

def compute_min_max(dataloader):
    min_value = float('inf')
    max_value = float('-inf')

    for _, samples, _ in dataloader:
        global_min_max.min = min(min_value, samples.min().item())
        global_min_max.max = max(max_value, samples.max().item())



def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """


    tb_log_dir_prefix = f'CNN_bn_bs={args.batch_size}_lr={args.learning_rate}_mom={args.momentum}_epochs={args.epochs}_length={args.length}_stride={args.stride}_extension={args.extension}_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    main(parser.parse_args())