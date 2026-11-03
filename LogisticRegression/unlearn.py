"""GPU version of unlearn_error.py.

All client datasets are moved to the GPU once, and every gradient/loss/accuracy
computation runs on the GPU using float64 (matching the numpy version numerically).
"""

from copy import deepcopy

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
    train_loaders, test_loader, backdoor_test_loader, client_indices, client_unlearn_indices, train_dataset = load_dataloader(args)
    client_remain_indices = [list(set(indices) - set(unlearn_indices)) for indices, unlearn_indices in zip(client_indices, client_unlearn_indices)]

    # unlearn with random labeling, since we only have two classes, we can simply flip the labels for the unlearned data (-1 to 1 or 1 to -1)
    for i, unlearn_indices in enumerate(client_unlearn_indices):
        for idx in unlearn_indices:
            # randomize/flip the label
            train_loaders[i].dataset.dataset.targets[idx] = 1 - train_loaders[i].dataset.dataset.targets[idx]

    Xs, ys = map(list, zip(*[dataloader_to_tensor(train_loader, device=device) for train_loader in train_loaders]))
    Xs_unlearn, ys_unlearn = map(list, zip(*[dataloader_to_tensor(DataLoader(Subset(train_dataset, indices=unlearn_indices), batch_size=args.batch_size, shuffle=False), device=device) for unlearn_indices in client_unlearn_indices]))

    Xs_remain, ys_remain = map(list, zip(*[dataloader_to_tensor(DataLoader(Subset(train_dataset, indices=remain_indices), batch_size=args.batch_size, shuffle=False), device=device) for remain_indices in client_remain_indices]))

    X_test, y_test = dataloader_to_tensor(test_loader, device=device)
    X_backdoor_test, y_backdoor_test = dataloader_to_tensor(backdoor_test_loader, device=device)

    # concatenate all clients' data to compute global metrics
    X_all, y_all = torch.cat(Xs, dim=0), torch.cat(ys, dim=0)
    X_unlearn_all, y_unlearn_all = torch.cat(Xs_unlearn, dim=0), torch.cat(ys_unlearn, dim=0)
    X_remain_all, y_remain_all = torch.cat(Xs_remain, dim=0), torch.cat(ys_remain, dim=0)

    lam_us = [Xs_unlearn[i].shape[0] / Xs[i].shape[0] for i in range(num_clients)]
    lam_rs = [Xs_remain[i].shape[0] / Xs[i].shape[0] for i in range(num_clients)]

    lr = args.lr
    lam = args.weight_decay
    epochs = args.global_epochs

    w = torch.tensor(np.load(f"saves/models/w_seed{args.seed}_alpha{args.alpha}.npy"), dtype=torch.float64, device=device)
    ws = [torch.zeros(Xs[0].shape[1], dtype=torch.float64, device=device) for _ in range(num_clients)]

    communicators = [eval(args.communicator)(top_k(k=1)) for _ in range(num_clients)]

    if args.communicator == "EF21Plus":
        args.communicator = "EF21+"

    if not os.path.exists(f"saves/ckpts/{args.communicator}_seed{args.seed}_alpha{args.alpha}"):
        os.makedirs(f"saves/ckpts/{args.communicator}_seed{args.seed}_alpha{args.alpha}")

    with torch.no_grad():
        test_acc = evaluate_accuracy(w, X_test, y_test)
        backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
        print(f"Epoch 0/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}")

    for epoch in range(epochs):
        with torch.no_grad():
            if is_save:
                np.save(f"saves/ckpts/{args.communicator}_seed{args.seed}_alpha{args.alpha}/{epoch}.npy", w.detach().cpu().numpy())
            grads = []
            for i in range(num_clients):
                ws[i] = w.clone()
                grad_u = grad_logistic(ws[i], Xs_unlearn[i], ys_unlearn[i], lam=lam)
                grad_r = grad_logistic(ws[i], Xs_remain[i], ys_remain[i], lam=lam)
                if args.lgp:
                    grad = conflict_reduce(grad_u, grad_r, lam=lam_us[i])
                else:
                    grad = grad_logistic(ws[i], Xs[i], ys[i], lam=lam)
                grads.append(grad)

            # aggregate
            grad = torch.stack(grads, dim=0).mean(dim=0)
            w = gradient_descent(w, grad, lr=lr)

            if (epoch + 1) % 1 == 0:
                test_acc = evaluate_accuracy(w, X_test, y_test)
                backdoor_test_acc = evaluate_accuracy(w, X_backdoor_test, y_backdoor_test)
                print(f"Epoch {epoch+1}/{epochs}, Test Accuracy: {test_acc:.4f}, Backdoor Test Accuracy: {backdoor_test_acc:.4f}")

    if is_save:
        np.save(f"saves/ckpts/{args.communicator}_seed{args.seed}_alpha{args.alpha}/{epochs}.npy", w.detach().cpu().numpy())
        np.save(f"saves/models/w_{args.communicator}_seed{args.seed}_alpha{args.alpha}.npy", w.detach().cpu().numpy())
