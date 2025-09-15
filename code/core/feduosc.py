"""
:file: feduosc.py
:date: 2025-08-23
:description: Federated Unlearning with Oriented Saliency Compression Algorithm 
:paper: Boxu Xiao, Sijia Liu, and Qing Ling, Federated unlearning with oriented saliency compression, in IJCNN 2025
"""


from typing import Optional, List, Dict, Any, Literal 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .federatedbase import Client, Server
from utils import Communicator


class ClientFedUOSC(Client): 
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Gradient Balancing(FedUOSC) Algorithm

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
        self.sparsity = 0.5 # default sparsity level for saliency compression

    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        super().set_dataloader(batch_size, shuffle)
        self.unlearning_indices = unlearning_indices
        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices)
        
        self.gamma = len(self.unlearning_indices) / len(self.data_indices) # strike the balance between unlearning and refinement loss

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

    def get_communication_cost(self, unit = 'MB') -> float:
        return self.comm.get_communication_cost(unit=unit) * self.sparsity

    @torch.no_grad()
    def communicate(self, global_model: nn.Module, mask: Dict[str, Any]) -> Dict[str, Any]: 
        """saliency compression"""
        local_state_dict = self.local_model.state_dict()
        self.comm(mask)
        global_state_dict = global_model.state_dict()

        for name in mask.keys():
            local_state_dict[name] = mask[name] * local_state_dict[name] + (1 - mask[name]) * global_state_dict[name]
        
        return local_state_dict

    def local_unlearning(self, loader: Optional[DataLoader] = None) -> None:
        """Local unlearning method for FedUOSC algorithm

        :param loader: Dataloader for unlearning, defaults to unlearning_loader
        """
        self.local_model.train()
        self.opt.zero_grad()
        if self.is_unlearner: 
            unlearn_data, unlearn_target = next(iter(self.unlearning_loader))
            unlearn_data, unlearn_target = unlearn_data.cuda(), unlearn_target.cuda()
            
            output = self.local_model(unlearn_data)
            loss = -self.gamma * self.criterion(output, unlearn_target)

            if self.remaining_indices: 
                remaining_data, remaining_target = next(iter(self.remaining_loader))
                remaining_data, remaining_target = remaining_data.cuda(), remaining_target.cuda()
                
                remaining_output = self.local_model(remaining_data)
                remaining_loss = self.criterion(remaining_output, remaining_target)
                
                loss += (1 - self.gamma) * remaining_loss

            loss.backward()

            if self.args.clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.clipping)

            self.opt.step()
        else: 
            remaining_data, remaining_target = next(iter(self.remaining_loader))
            remaining_data, remaining_target = remaining_data.cuda(), remaining_target.cuda()
            remaining_output = self.local_model(remaining_data)
            loss = self.criterion(remaining_output, remaining_target) 
            loss.backward()

            if self.args.clipping > 0:
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.args.clipping)

            self.opt.step()

    def local_refinement(self, loader: Optional[DataLoader] = None) -> None:
        """Local refinement method for FedUOSC algorithm

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


class ServerFedUOSC(Server): 
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for FedUOSC Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args)
        self.sparsity = 0.5

    def get_saliency_mask(self, loader: DataLoader) -> Dict[str, Any]:
        """Generate saliency mask based on the gradients from the unlearning data

        :param loader: Dataloader for unlearning data
        :return: Saliency mask
        """
        self.global_model.eval()
        self.global_model.zero_grad()

        total = 0
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = self.global_model(data)
            loss = self.criterion(output, target)
            loss.backward()
            total += 1

        saliency = {}
        for name, param in self.global_model.named_parameters(): 
            saliency[name] = param.grad.abs().clone()

        mask = {}
        for name, sal in saliency.items():
            numel = sal.numel()
            k = int(self.sparsity * numel)
            if k < 1:
                mask[name] = torch.zeros_like(sal)
            else:
                flat_sal = sal.view(-1)
            if k >= numel:
                threshold = flat_sal.min() - 1 
            else:
                threshold = torch.topk(flat_sal, k, largest=True).values[-1]
            mask[name] = (sal >= threshold).float()

        return mask

    @torch.no_grad()
    def aggregate(self, clients: List[ClientFedUOSC], mask: Dict[str, Any]) -> None: 
        """Aggregate client models

        :param clients: list of Client
        """
        uploads = [client.communicate(self.global_model, mask) for client in clients]
        global_state_dict = self.global_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.mean(
                torch.stack([upload[key].float() for upload in uploads]), dim=0
            )
        self.global_model.load_state_dict(global_state_dict)

    def __call__(self, clients: List[ClientFedUOSC], args=None) -> nn.Module:
        """Run Federated Learning with FedUOSC algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            acc = self.evaluate(args.test_loader)
            backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        # Unlearning Phase0: Saliency Compression Mask Generation
        unlearning_mask = self.get_saliency_mask(
            DataLoader(
                Subset(clients[0].global_dataset, [i for client in clients for i in client.unlearning_indices if client.is_unlearner]), 
                batch_size=args.batch_size, 
                shuffle=False, 
            )
        )

        remaining_mask = self.get_saliency_mask(
            DataLoader(
                Subset(clients[0].global_dataset, [i for client in clients for i in client.remaining_indices if client.join_refine]), 
                batch_size=args.batch_size, 
                shuffle=False, 
            )
        )

        # Unlearning Phase1: Balanced Forgetting 
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        temp_epoch = 0
        
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients, unlearning_mask)

            if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

                if backdoor_acc < 0.005: 
                    temp_epoch = epoch
                    break
        
        for epoch in range(5): 
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning() 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients, unlearning_mask)

            if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + temp_epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")

        args.lr = args.ft_lr if args.ft_lr else args.lr

        for client in clients: 
            if client.join_refine:
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
            self.aggregate(refine_clients, remaining_mask)


            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)
                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}")


        return self.global_model

