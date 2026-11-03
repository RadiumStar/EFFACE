"""GPU (PyTorch) implementations of compressors, communicators and logistic-regression primitives.

This is the torch/GPU counterpart of optimizer.py. All operations run on the device of the
input tensors (CUDA when available). The numerical logic is identical to the numpy version.
"""

import torch

from utils import *


# compressors

class top_k:
    def __init__(self, k):
        self.k = k

    def __call__(self, grad):
        if self.k >= grad.numel():
            return grad
        sparse_grad = torch.zeros_like(grad)
        # torch.topk selects the k largest |values| by magnitude
        indices = torch.topk(grad.abs(), self.k).indices
        sparse_grad[indices] = grad[indices]
        return sparse_grad


# communicators

class Uncompressed:
    def __init__(self, compressor):
        pass

    def __call__(self, grad):
        return grad


class Compressed:
    def __init__(self, compressor):
        self.compressor = compressor

    def __call__(self, grad):
        return self.compressor(grad)


class EF14:
    def __init__(self, compressor):
        self.compressor = compressor
        self.error = None

    def __call__(self, grad):
        if self.error is None:
            self.error = torch.zeros_like(grad)
            return grad
        grad = grad + self.error
        compressed_grad = self.compressor(grad)
        self.error = grad - compressed_grad
        return compressed_grad


class EF21:
    def __init__(self, compressor):
        self.compressor = compressor
        self.prev = None

    def __call__(self, grad):
        if self.prev is None:
            self.prev = grad.clone()
            return grad
        diff = grad - self.prev
        compressed_diff = self.compressor(diff)
        self.prev = self.prev + compressed_diff
        return self.prev


class EF21Plus:
    def __init__(self, compressor):
        self.compressor = compressor
        self.prev = None

    def __call__(self, grad):
        if self.prev is None:
            self.prev = grad.clone()
            return grad

        phi = self.compressor(grad)
        psi = self.prev + self.compressor(grad - self.prev)

        PHI = torch.sum((grad - phi) ** 2)
        PSI = torch.sum((grad - psi) ** 2)

        if PHI <= PSI:
            self.prev = phi
            return phi.clone()
        else:
            self.prev = psi
            return psi.clone()


class EControl:
    def __init__(self, compressor, eta=0.0001):
        self.compressor = compressor
        self.eta = eta
        self.prev = None
        self.error = None

    def __call__(self, grad):
        if self.prev is None:
            self.prev = grad.clone()
            self.error = torch.zeros_like(grad)
            return grad

        diff = grad - self.prev
        compressed_diff = self.compressor(self.eta * self.error + diff)
        self.prev = self.prev + compressed_diff
        self.error = self.error + diff - compressed_diff
        return self.prev.clone()


class EFFACE:
    def __init__(self, compressor, eta=0.0001):
        self.compressor = compressor
        self.eta = eta
        self.prev = None
        self.error = None

    def __call__(self, grad):
        if self.prev is None:
            self.prev = grad.clone()
            self.error = torch.zeros_like(grad)
            return grad

        phi = self.compressor(self.eta * self.error + grad)
        psi = self.prev + self.compressor(self.eta * self.error + grad - self.prev)

        PHI = torch.sum((self.eta * self.error + grad - phi) ** 2)
        PSI = torch.sum((self.eta * self.error + grad - psi) ** 2)

        if PHI <= PSI:
            self.prev = phi
            self.error = self.error + grad - phi
            # print("GradComp")
            return phi.clone()
        else:
            self.prev = psi
            self.error = self.error + grad - psi
            # print("GradDiffComp")
            return psi.clone()


def gradient_descent(w, grad, lr=0.01, communicator=Uncompressed(top_k(k=1))):
    """Performs a single gradient descent step."""
    compressed_grad = communicator(grad)
    return w - lr * compressed_grad


def logistic_loss(w, X, y, lam=0.0):
    """Logistic loss: L(w) = (1/n) Σ log(1 + exp(-y_i w^T x_i)) + (λ/2) ||w||²

    :param lam: regularization parameter, 0.0 for convex case, >0 for strong convex case
    """
    z = y * (X @ w)
    log_exp = torch.log1p(torch.exp(-z))
    loss_unreg = torch.mean(log_exp)
    reg = 0.5 * lam * torch.sum(w ** 2)
    return (loss_unreg + reg).item()


def grad_logistic(w, X, y, lam=0.0):
    """Gradient of logistic loss: ∇L(w) = -(1/n) Σ [y_i σ(-y_i w^T x_i) x_i] + λ w

    :param lam: regularization parameter, 0.0 for convex case, >0 for strong convex case
    """
    z = y * (X @ w)
    sigma_negz = 1.0 / (1.0 + torch.exp(z))  # σ(-z)
    grad_unreg = -(X.T @ (y * sigma_negz)) / X.shape[0]
    grad_reg = lam * w
    return grad_unreg + grad_reg


def evaluate_accuracy(w, X, y):
    """Evaluate classification accuracy."""
    logits = X @ w
    preds = (logits >= 0).to(torch.int64)  # predict +1 if logit >= 0 else 0

    y_binary = ((y + 1) // 2).to(torch.int64)  # convert from {-1, +1} to {0, 1}
    accuracy = torch.mean((preds == y_binary).to(torch.float64))
    return accuracy.item()


def conflict_reduce(grad_u, grad_r, lam=0.001):
    """If cosine similarity between grad_u and grad_r is negative, return the projection of
    grad_u onto the normal plane of grad_r, else return lam * grad_u + (1 - lam) * grad_r.

    :param grad_u: gradient on unlearning dataset
    :param grad_r: gradient on remaining dataset
    :param lam: weight of unlearning term
    """
    cos_sim = cosine_similarity(grad_u, grad_r)
    if cos_sim < 0:
        grad_u_proj = grad_u - (torch.dot(grad_u, grad_r) / torch.dot(grad_r, grad_r)) * grad_r
        return lam * grad_u_proj + (1 - lam) * grad_r
    else:
        return lam * grad_u + (1 - lam) * grad_r
