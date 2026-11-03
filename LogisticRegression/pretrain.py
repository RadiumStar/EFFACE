"""GPU version of pretrain.py.

All client datasets are moved to the GPU once, and every gradient/loss/accuracy
computation runs on the GPU using float64 (matching the numpy version numerically).
"""

import os

import numpy as np
import torch

from init import get_args
from utils import *
from optimizer import *
from loader import *


if __name__ == "__main__":
    args = get_args("config/config.yml")
    lr = args.lr
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device_idx = args.cuda
    device = torch.device('cuda:{}'.format(device_idx) if torch.cuda.is_available() and device_idx != -1 else 'cpu') 
    print(f">>> Using device: {device}")

    is_save = args.save_trajectory

    num_clients = args.num_clients

    # load data
    train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, _ = load_dataloader(args)

    Xs, ys = map(list, zip(*[dataloader_to_tensor(train_loader, device=device) for train_loader in train_loaders]))

    X_test, y_test = dataloader_to_tensor(test_loader, device=device)
    X_backdoor_test, y_backdoor_test = dataloader_to_tensor(backdoor_test_loader, device=device)

    lr = args.lr
    lam = args.weight_decay
    epochs = args.global_epochs

    w = torch.zeros(Xs[0].shape[1], dtype=torch.float64, device=device)
    ws = [torch.zeros(Xs[0].shape[1], dtype=torch.float64, device=device) for _ in range(num_clients)]

    if is_save:
        os.makedirs(f"saves/ckpts/pretrain_seed{args.seed}_alpha{args.alpha}", exist_ok=True)

    with torch.no_grad():
        test_acc = evaluate_accuracy(w, X_test, y_test)
        backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
        print(f"Epoch 0/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}")

    for epoch in range(epochs):
        with torch.no_grad():
            if is_save:
                np.save(f"saves/ckpts/pretrain_seed{args.seed}_alpha{args.alpha}/{epoch}.npy", w.detach().cpu().numpy())

            grads = []
            for i in range(num_clients):
                ws[i] = w.clone()
                grads.append(grad_logistic(ws[i], Xs[i], ys[i], lam=lam))

            # aggregate
            grad = torch.stack(grads, dim=0).mean(dim=0)
            w = gradient_descent(w, grad, lr=lr)

            if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
                test_acc = evaluate_accuracy(w, X_test, y_test)
                backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
                print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}")

    if is_save:
        np.save(f"saves/ckpts/pretrain_seed{args.seed}_alpha{args.alpha}/{epochs}.npy", w.detach().cpu().numpy())

    os.makedirs("saves/models", exist_ok=True)
    np.save(f"saves/models/w_seed{args.seed}_alpha{args.alpha}.npy", w.detach().cpu().numpy())
