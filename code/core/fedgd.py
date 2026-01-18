"""
:file: fedgb.py
:date: 2025-08-05
:description: Federated Gradient Descent Algorithm for Unlearning 
"""


from typing import Optional, List 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator, save_checkpoint

class ClientFedGD(Client): 
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
        
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.opt,
        #     milestones=getattr(self.args, 'milestones', [100000]),
        #     gamma=getattr(self.args, 'gamma', 0.1)
        # )

    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)
        self.unlearning_indices = unlearning_indices
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)

        self.unlearning_loader = DataLoader(
            Subset(self.global_dataset, self.unlearning_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.is_unlearner else None
        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        return self.local_refinement(loader)

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedGD algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        if loader is None:
            loader = self.remaining_loader
        
        self.local_model.train()
        
        # for data, target in loader:
        #     data, target = data.cuda(), target.cuda()
        #     self.opt.zero_grad()
        #     output = self.local_model(data)
        #     loss = self.criterion(output, target)
        #     loss.backward()
        #     self.opt.step()
        # self.scheduler.step()

        data, target = next(iter(loader))
        data, target = data.cuda(), target.cuda()
        self.opt.zero_grad()
        output = self.local_model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.opt.step()
        # self.scheduler.step()

    def weight_decay_schedule(self, epoch: int) -> None: 
        if epoch % 150 == 0: 
            for param_group in self.opt.param_groups:
                param_group['weight_decay'] *= 0.5


class ServerFedGD(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedGD Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)

    def __call__(self, clients: List[ClientFedGD], args=None) -> nn.Module:
        """Run Federated Learning with FedGD algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        # [note] make it more flexible
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 100)
        
        # while epoch < unlearn_epochs:
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_refine:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 
                    client.weight_decay_schedule(epoch + 1)

            unlearn_clients = [
                client for client in clients if client.join_refine
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0: 
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")
                

        args.lr = args.ft_lr if args.ft_lr else args.lr

        return self.global_model
        