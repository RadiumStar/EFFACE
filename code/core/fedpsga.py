"""
:file: fedpsga.py
:date: 2025-08-21
:description: Federated Projected Stochastic Gradient Ascent Algorithm for Unlearning 
:paper: Anisa Halimi, Swanand Kadhe, Ambrish Rawat, and Nathalie Baracaldo, Federated unlearning: How to efficiently erase a client in FL?,” arXiv preprint arXiv: 2207.05521, 2022. 
:note: It is worth noting that the original paper only provide an unlearning method for client removal. Here we regard the unlearning dataset as a special client and do federated projected stochastic gradient ascent on it, followed by a global model refinement on the remaining clients.
"""


from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from models import get_model_class
from utils import Communicator
from utils.misc import get_distance


class ClientFedPSGA(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Projected Stochastic Gradient Ascent(FedPSGA) Algorithm

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

        self.clip_grad = 1
        self.distance_threshold = 2.2 

        model_class = get_model_class(args.model)
        with torch.no_grad():
            self.reference_model = model_class()
            self.reference_model.load_state_dict(self.local_model.state_dict())
            self.reference_model.cuda()
            self.reference_model.eval()

        dist_ref_random_lst = [get_distance(self.reference_model, model_class().cuda()) for _ in range(10)] 
        self.radius = np.mean(dist_ref_random_lst) / 3 
        # print(f"Reference model radius: {self.radius:.4f}")

    def set_dataloader(self, batch_size=64, shuffle=True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)
        self.unlearning_indices = unlearning_indices
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = bool(self.unlearning_indices) 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = self.join_unlearning

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
        """Local unlearning method for FedPSGA algorithm"""
        self.local_model.train()

        unlearn_data, unlearn_target = next(iter(self.unlearning_loader))
        unlearn_data, unlearn_target = unlearn_data.cuda(), unlearn_target.cuda()

        self.opt.zero_grad()
        output = self.local_model(unlearn_data)
        loss = -self.criterion(output, unlearn_target)
        loss.backward()

        if self.clip_grad > 0: 
            nn.utils.clip_grad_norm_(self.local_model.parameters(), self.clip_grad)

        self.opt.step()

        # Project the model parameters to the reference model's radius
        with torch.no_grad():
            dist = get_distance(self.local_model, self.reference_model)
            if dist > self.radius:  # Project 
                dist_vec = nn.utils.parameters_to_vector(self.local_model.parameters()) - nn.utils.parameters_to_vector(self.reference_model.parameters())
                dist_vec = dist_vec / (dist ** 2) * np.sqrt(self.radius) 
                proj_vec = nn.utils.parameters_to_vector(self.reference_model.parameters()) + dist_vec
                nn.utils.vector_to_parameters(proj_vec, self.local_model.parameters())
            
            # if get_distance(self.local_model, self.reference_model) > self.distance_threshold:
            #     stop = True

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedPSGA algorithm

        :param loader: Dataloader for refinement, defaults to remaining_loader
        """
        if loader is None:
            loader = self.remaining_loader
        
        self.local_model.train()
        
        data, target = next(iter(loader))
        data, target = data.cuda(), target.cuda()
        self.opt.zero_grad()
        output = self.local_model(data)
        loss = self.criterion(output, target)
        loss.backward()

        self.opt.step()


class ServerFedPSGA(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for Federated Projected Stochastic Gradient Ascent(FedPSGA) Algorithm

        :param global_model: Global model
        :param comm: Communicator for communication between client and server, defaults to Communicator()
        :param args: Other arguments
        """
        super().__init__(global_model, args=args)

    def __call__(self, clients: List[ClientFedPSGA], args=None) -> nn.Module:
        """Run Federated Learning with FedGB algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        # Unlearning Phase1: Project Stochastic Gradient Ascent
        for epoch in range(getattr(self.args, 'unlearn_epochs', 2)):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

        unlearn_clients = [
            client for client in clients if client.join_unlearning
        ]

        self.aggregate(unlearn_clients)

        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch 0, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        args.lr = args.ft_lr if args.ft_lr else args.lr

        for client in clients: 
            if client.join_refine:
                client.set_optimizer(args)
                client.comm.reset()

        # Unlearning Phase2: Post Training
        for epoch in range(getattr(self.args, 'refine_epochs', 20)):
            for client in clients:
                if client.join_refine:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_refinement()

            refine_clients = [
                client for client in clients if client.join_refine
            ]
            self.aggregate(refine_clients)


            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                    acc = self.evaluate(args.test_loader)
                    backdoor_acc = self.evaluate(args.backdoor_test_loader)
                    print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        return self.global_model