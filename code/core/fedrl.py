"""
:file: fedrl.py
:date: 2025-07-29
:description: Federated Random Labeling Algorithm for Unlearning
"""

from typing import Optional, List
import random 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator, save_checkpoint


class ClientFedRL(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Random Labeling(FedRL) Algorithm

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

    def set_dataloader(self, batch_size=64, shuffle=True, unlearning_indices: List[int] = []) -> None:
        self.unlearning_indices = unlearning_indices

        # set random labels for local unlearning datasets
        if self.args.dataset not in ['mnist']:
            for idx in self.unlearning_indices: 
                new_labels = list(range(self.args.num_classes)) 
                new_labels.remove(int(self.global_dataset.dataset.targets[idx]))
                if isinstance(self.global_dataset.dataset.targets, torch.Tensor):
                    self.global_dataset.dataset.targets[idx] = torch.tensor(random.choice(new_labels), dtype=self.global_dataset.dataset.targets.dtype)
                else:
                    self.global_dataset.dataset.targets[idx] = random.choice(new_labels)
        else:
            for idx in self.unlearning_indices: 
                self.global_dataset.dataset.targets[idx] = self.global_dataset.base_dataset.targets[idx]

        super().set_dataloader(batch_size, shuffle)

        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)
        # self.gamma = 0.5
        self.gamma = len(self.unlearning_indices) / len(self.data_indices) # strike the balance between unlearning and refinement loss

        self.unlearning_loader = DataLoader(
            Subset(self.global_dataset, self.unlearning_indices), 
            batch_size=batch_size, 
            shuffle=shuffle
        ) if self.is_unlearner else None
        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=batch_size, 
            shuffle=shuffle
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method for FedRL algorithm

        :param loader: Dataloader for unlearning, defaults to unlearning_loader
        """
        self.local_model.train()
        self.opt.zero_grad()
        
        data, target = next(iter(self.train_loader)) 
        data, target = data.cuda(), target.cuda()

        output = self.local_model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.opt.step()

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedGB algorithm

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

        if self.args.clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.clipping)

        self.opt.step()


class ServerFedRL(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedGB Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)

    def __call__(self, clients: List[ClientFedRL], args=None) -> nn.Module:
        """Run Federated Learning with FedGB algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

            if args.check:
                params = torch.cat([p.data.view(-1) for p in self.global_model.parameters()]).cpu().clone()
                state = {'params': params}
                if args.communicator == 'Communicator': 
                    if args.compressor == 'Compressor': 
                        save_checkpoint(state, path=f"{0}", folder=f"saves/checkpoints/cifar10/Uncompressed")
                    else:
                        save_checkpoint(state, path=f"{0}", folder=f"saves/checkpoints/cifar10/Compressed_{args.k}")
                else:
                    save_checkpoint(state, path=f"{0}", folder=f"saves/checkpoints/cifar10/{args.communicator}fix_{args.k}")

        # Unlearning Phase1: Balanced Forgetting 
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")
                if args.check:
                    params = torch.cat([p.data.view(-1) for p in self.global_model.parameters()]).cpu().clone()
                    state = {'params': params}
                    if args.communicator == 'Communicator': 
                        if args.compressor == 'Compressor': 
                            save_checkpoint(state, path=f"{epoch+1}", folder=f"saves/checkpoints/cifar10/Uncompressed")
                        else:
                            save_checkpoint(state, path=f"{epoch+1}", folder=f"saves/checkpoints/cifar10/Compressed_{args.k}")
                    else:
                        save_checkpoint(state, path=f"{epoch+1}", folder=f"saves/checkpoints/cifar10/{args.communicator}fix_{args.k}")

        args.lr = args.ft_lr if args.ft_lr else args.lr

        for client in clients: 
            if client.join_refine:
                args.weight_decay = 0.01
                client.set_optimizer(args)
                client.comm.reset()

        # Unlearning Phase2: Utility Refinement
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


            if (epoch + 1) % 1 == 0: 
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {unlearn_epochs + epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")
                if args.check:
                    params = torch.cat([p.data.view(-1) for p in self.global_model.parameters()]).cpu().clone()
                    state = {'params': params}
                    save_checkpoint(state, path=f"{epoch+unlearn_epochs+1}", folder=f"saves/checkpoints/cifar10/{args.communicator}")

        return self.global_model
        