""" 
:file: fedrl_gru.py
:date: 2026-10-30
:description: Federated Unlearning using Random Labeling with Gradient Rectified Unlearning
:src: [GRU: Mitigating the Trade-off between Unlearning and Retention for LLMs (ICML 2025)](https://arxiv.org/pdf/2503.09117)
"""
 
from copy import deepcopy
import random
import time
from typing import Optional, List 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np 
from torch.cuda.amp import autocast, GradScaler

from .federatedbase import Client, Server
from utils import Communicator, save_checkpoint, save_model


class ClientFedRLGRU(Client):
    def __init__(self, global_dataset: Dataset, data_indices: List[int], local_model: nn.Module, client_id: int = 0, comm: Communicator = Communicator(), unlearning_indices: List[int] = [], args=None):
        """Federated Learning Client for Federated Random Labeling with Gradient Projection(FedRLGRU) Algorithm
        
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

        self.cos_threshold = getattr(self.args, 'cos_threshold', -1)


    def set_dataloader(self, batch_size = 64, shuffle = True, unlearning_indices: List[int] = []) -> None:
        self.batch_size = batch_size
        self.unlearning_indices = unlearning_indices

        for idx in self.unlearning_indices: 
            new_labels = list(range(self.args.num_classes)) 
            new_labels.remove(int(self.global_dataset.dataset.targets[idx]))
            if isinstance(self.global_dataset.dataset.targets, torch.Tensor):
                self.global_dataset.dataset.targets[idx] = torch.tensor(random.choice(new_labels), dtype=self.global_dataset.dataset.targets.dtype)
            else:
                self.global_dataset.dataset.targets[idx] = random.choice(new_labels)

        super().set_dataloader(batch_size, shuffle)

        self.remaining_indices = list(set(self.data_indices) - set(unlearning_indices))

        self.join_unlearning = True 
        self.join_refine = bool(self.remaining_indices)
        self.is_unlearner = bool(self.unlearning_indices) 
        self.gamma = len(self.unlearning_indices) / len(self.data_indices) # strike the balance between unlearning and refinement loss
        self.unlearning_batch_size = int(self.gamma * self.batch_size) 
        self.remaining_batch_size = self.batch_size - self.unlearning_batch_size
        self.unlearning_loader = DataLoader(
            Subset(self.global_dataset, self.unlearning_indices), 
            batch_size=self.unlearning_batch_size, 
            shuffle=shuffle, 
        ) if self.is_unlearner else None
        self.remaining_loader = DataLoader(
            Subset(self.global_dataset, self.remaining_indices), 
            batch_size=self.remaining_batch_size, 
            shuffle=shuffle, 
        ) if self.join_refine else None

    def local_unlearning(self, loader: Optional[DataLoader] = None, epoch: int = 0) -> None:
        """Local unlearning method for FedRLGRU

        :param loader: Dataloader for unlearning, defaults to None
        :param epoch: Current epoch, defaults to 0
        """
        self.local_model.train()
        self.opt.zero_grad()

        # scaler = GradScaler() if self.args.dataset in ['tiny_imagenet', 'imagenet100'] else None

        if self.is_unlearner:  
            # gamma = self.unlearning_batch_size / self.batch_size
            gamma = self.gamma

            unlearn_data, unlearn_target = next(iter(self.unlearning_loader))
            unlearn_data, unlearn_target = unlearn_data.cuda(), unlearn_target.cuda()

            remain_data, remain_target = next(iter(self.remaining_loader))
            remain_data, remain_target = remain_data.cuda(), remain_target.cuda()

            # mix unlearning and remaining data into one batch
            data = torch.cat([unlearn_data, remain_data], dim=0)
            target = torch.cat([unlearn_target, remain_target], dim=0) 

            output = self.local_model(data)
            output_u, output_r = output[:self.unlearning_batch_size], output[self.unlearning_batch_size:]

            loss = self.criterion(output, target) 

            loss_u, loss_r = self.criterion(output_u, unlearn_target), self.criterion(output_r, remain_target) 

            grad_u = torch.autograd.grad(loss_u, self.local_model.parameters(), retain_graph=True)
            grad_r = torch.autograd.grad(loss_r, self.local_model.parameters(), retain_graph=True) 
            
            cos_threshold = self.cos_threshold 
            cos_sims = []
            final_grads = [] 

            g_u_flat = torch.cat([g_u.detach().view(-1) for g_u in grad_u])
            g_r_flat = torch.cat([g_r.detach().view(-1) for g_r in grad_r])

            dot_product = torch.dot(g_u_flat, g_r_flat)
            norm_u = torch.norm(g_u_flat)
            norm_r = torch.norm(g_r_flat)

            cos_sim = dot_product / (norm_u * norm_r + 1e-8)
            cos_sims.append(cos_sim.item())

            norm_r_sq = torch.dot(g_r_flat, g_r_flat) + 1e-8
            g_u_flat = g_u_flat - (dot_product / norm_r_sq) * g_r_flat
                
            norm_u = torch.norm(g_u_flat)
            if norm_u * gamma > norm_r * (1 - gamma):
                g_u_flat = g_u_flat * (norm_r * (1 - gamma) / (norm_u * gamma + 1e-8))
                
            combined_grad = g_u_flat    # [NOTE] key difference between ours and GRU, where GRU directly uses g_u_flat as the final gradient

            # Split the combined gradient back into the original parameter shapes
            final_grads = []
            start_idx = 0
            for p in self.local_model.parameters():
                param_size = p.numel()
                final_grads.append(combined_grad[start_idx:start_idx + param_size].view(p.shape))
                start_idx += param_size 

            self.opt.step() 
        else: 
            remaining_data, remaining_target = next(iter(self.remaining_loader))
            remaining_data, remaining_target = remaining_data.cuda(), remaining_target.cuda()
            remaining_output = self.local_model(remaining_data)
            loss = self.criterion(remaining_output, remaining_target) 
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


class ServerFedRLGRU(Server):
    def __init__(self, global_model: nn.Module, args=None):
        """Federated Learning Server for Federated Random Labeling with Gradient Projection(FedRLGRU) Algorithm

        :param global_model: Global model for the server
        :param args: Other arguments
        """
        super().__init__(global_model, args=args)

    def __call__(self, clients: List[torch.Tensor], args=None) -> nn.Module: 
        """Run Federated Learning with FedRL algorithm

        :param clients: List of clients participating in the federated learning
        :param args: Other arguments
        :return: Updated global model after federated learning
        """
        backdoor_acc_threshold = getattr(self.args, 'unlearn_acc_threshold', 1 / self.args.num_classes)
        first_in_threshold = True
        print(">>> Backdoor accuracy threshold for unlearning: {:.4f}".format(backdoor_acc_threshold))
        best_acc = 0.0
        best_unlearn_acc = 1.0
        best_epoch = 0
        best_model_state = None

        if args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
            best_acc = self.evaluate(args.test_loader)
            best_backdoor_acc = self.evaluate(args.backdoor_test_loader)
            print(f"Global Epoch {0}, Accuracy: {best_acc:.4f}, Backdoor Test Accuracy: {best_backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_backdoor_acc:.4f}") 

        start_time = time.time()
        start_comm = 0.0
        best_time, best_comm = 0.0, 0.0     # record the time and communication cost for the best model
        
        unlearn_epochs = getattr(self.args, 'unlearn_epochs', 80)
        
        for epoch in range(unlearn_epochs):
            for client in clients: 
                if client.join_unlearning:
                    client.set_model(self.global_model)
                    for _ in range(getattr(self.args, 'local_epochs', 1)):
                        client.local_unlearning(epoch=epoch) 

            unlearn_clients = [
                client for client in clients if client.join_unlearning
            ]

            self.aggregate(unlearn_clients)

            if (epoch + 1) % 1 == 0 and args and getattr(self.args, 'test_loader') and getattr(self.args, 'backdoor_test_loader'):
                acc = self.evaluate(args.test_loader)
                backdoor_acc = self.evaluate(args.backdoor_test_loader)

                if backdoor_acc <= backdoor_acc_threshold: 
                    if acc >= best_acc or first_in_threshold: 
                        first_in_threshold = False
                        best_acc = acc
                        best_unlearn_acc = backdoor_acc
                        best_epoch = epoch + 1
                        best_model_state = deepcopy(self.global_model.state_dict())
                        best_time = time.time() - start_time
                        best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                        if args.save: 
                            save_model(best_model_state, args.save)
                elif backdoor_acc <= best_unlearn_acc: 
                    best_acc = acc
                    best_unlearn_acc = backdoor_acc
                    best_epoch = epoch + 1
                    best_model_state = deepcopy(self.global_model.state_dict())
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients])
                    if args.save: 
                        save_model(best_model_state, args.save) 

                print(f"Global Epoch {epoch + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")

        args.lr = args.ft_lr if args.ft_lr else args.lr

        if getattr(self.args, 'refine_epochs', 20) > 0:
            start_time += time.time() - (best_time + start_time)  # reset the start time for refinement phase
            start_comm += sum([client.get_communication_cost(unit='MB') for client in clients]) - best_comm  # reset the start communication cost for refinement phase
            if best_model_state is not None:
                self.global_model.load_state_dict(best_model_state)

        for client in clients: 
            if client.join_refine: 
                client.set_optimizer(args)
                client.comm.reset()

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

                if backdoor_acc <= backdoor_acc_threshold: 
                    if acc >= best_acc or first_in_threshold: 
                        first_in_threshold = False
                        best_acc = acc
                        best_unlearn_acc = backdoor_acc
                        best_epoch = epoch + 1 + unlearn_epochs
                        best_model_state = deepcopy(self.global_model.state_dict())
                        best_time = time.time() - start_time
                        best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
                        if args.save: 
                            save_model(best_model_state, args.save)
                elif backdoor_acc <= best_unlearn_acc: 
                    best_acc = acc
                    best_unlearn_acc = backdoor_acc
                    best_epoch = epoch + 1 + unlearn_epochs
                    best_model_state = deepcopy(self.global_model.state_dict())
                    best_time = time.time() - start_time
                    best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
                    if args.save: 
                        save_model(best_model_state, args.save) 

                print(f"Global Epoch {epoch + unlearn_epochs + 1}, Accuracy: {acc:.4f}, Backdoor Test Accuracy: {backdoor_acc:.4f}, Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")

        if best_model_state is not None:
            self.global_model.load_state_dict(best_model_state)
        else:
            best_time = time.time() - start_time
            best_comm = sum([client.get_communication_cost(unit='MB') for client in clients]) - start_comm
        
        print(f"\n>>> Best Epoch: {best_epoch}, Best Accuracy: {best_acc:.4f}, Best Backdoor Accuracy: {best_unlearn_acc:.4f}")
        print(f"\n>>> Best time cost: {best_time:.4f} s")
        print(f">>> Best communication cost: {best_comm:.4f} MB\n")

        return self.global_model