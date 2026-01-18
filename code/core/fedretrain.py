"""
:file: fedretrain.py
:date: 2025-07-16
:description: Federated Retraining for Unlearning
"""


from typing import Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from models import get_model_class
from utils import Communicator


class ClientFedRetrain(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Gradient Balancing(FedGB) Algorithm

        :param global_dataset: Global dataset for the client
        :param data_indices: Indices of the local dataset for the client
        :param local_model: Local model
        :param client_id: Client index, defaults to 0
        :param comm: Communicator for communication between client and server, defaults to Communicator()
        :param args: Other arguments
        """
        super().__init__(global_dataset, data_indices, local_model, client_id, comm, args=args)

        self.set_dataloader(
            batch_size=getattr(self.args, 'batch_size', len(self.data_indices)), 
            shuffle=getattr(self.args, 'shuffle', True), 
            unlearning_indices=unlearning_indices
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=getattr(self.args, 'milestones', [500]),
            gamma=getattr(self.args, 'gamma', 0.1)
        )

    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)
        self.unlearning_indices = unlearning_indices
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)

        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method for FedGB algorithm

        :param loader: Dataloader for unlearning, defaults to unlearning_loader
        """
        if self.join_refine: 
            loader = self.remaining_loader
            
            self.local_model.train()
            for epoch in range(getattr(self.args, 'local_epochs', 1)):
                for data, target in loader:
                    data, target = data.cuda(), target.cuda()
                    self.opt.zero_grad()
                    output = self.local_model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.opt.step()
            self.scheduler.step()

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedGB algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        return self.local_unlearning(loader)


class ServerFedRetrain(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedGB Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)
        # reinitialize the global model as a random one
        model_class = get_model_class(args.model)
        self.global_model = model_class()
        self.global_model.cuda()

    def __call__(self, clients: List[ClientFedRetrain], args=None) -> nn.Module:
        """Run Federated Learning with FedGB algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        for epoch in range(getattr(self.args, 'global_epochs', 100)):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'unlearn_local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 100 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                # rem_acc, rem_loss = self.evaluate(args.remaining_loader, self.criterion)
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        return self.global_model
        