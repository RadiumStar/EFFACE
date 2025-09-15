"""
:file: fedavg.py
:date: 2025-07-11
:description: Federated Averaging Algorithm Implementation(Pretrained Model for Unlearning)
"""

from typing import List

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset 

from .federatedbase import Client, Server
from utils import Communicator

class ClientFedAvg(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None): 
        """Federated Learning Client for FedAvg Algorithm

        :param global_dataset: Global dataset for the client
        :param data_indices: Indices of the local dataset for the client
        :param local_model: Local model
        :param client_id: Client index, defaults to 0
        :param comm: Communicator for the client, defaults to Communicator()
        :param unlearning_indices: useless in FedAvg
        :param args: Other arguments
        """
        super().__init__(global_dataset, data_indices, local_model, client_id, comm, args=args)

        self.set_dataloader(
            batch_size=getattr(self.args, 'batch_size', len(self.local_dataset)), 
            shuffle=getattr(self.args, 'shuffle', True)
        )

    def train(self, loader: DataLoader = None) -> None:
        """Train local model using FedAvg algorithm

        :param loader: Dataloader for training, defaults to train_loader
        """
        if loader is None:
            loader = self.train_loader
        
        self.local_model.train()
        for epoch in range(getattr(self.args, 'local_epochs', 1)):
            for data, target in loader:
                data, target = data.cuda(), target.cuda()
                self.opt.zero_grad()
                output = self.local_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.opt.step()
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
    

class ServerFedAvg(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedAvg Algorithm

        :param global_model: Global model for the server 
        :param args: Other arguments
        """
        super().__init__(global_model, args)

    def __call__(self, clients: list[ClientFedAvg], args=None) -> nn.Module:
        """Run Federated Learning with FedAvg algorithm

        :param clients: List of client instances
        :param args: Additional arguments for training, defaults to None
        :return: Updated global model
        """
        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        for epoch in range(getattr(self.args, 'global_epochs', 200)):
            for client in clients:
                client.set_model(self.global_model)
                client.train()
            
            self.aggregate(clients)

            if (epoch + 1) % 100 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")
        
        return self.global_model
    