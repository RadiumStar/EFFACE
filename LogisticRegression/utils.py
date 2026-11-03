"""GPU (PyTorch) implementations of the utility functions used for distributed logistic regression.

These mirror utils.py but operate on torch tensors so that all heavy linear-algebra
(X @ w, X.T @ ..., element-wise ops) runs on the GPU. Use the same dtype (default float64)
to keep numerical results identical to the numpy version.
"""

import torch

DTYPE = torch.float64


def cosine_similarity(vec1, vec2):
    """Cosine similarity between two torch vectors, returned as a python float."""
    with torch.no_grad():
        dot_product = torch.dot(vec1, vec2)
        norm_vec1 = torch.linalg.norm(vec1)
        norm_vec2 = torch.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return (dot_product / (norm_vec1 * norm_vec2)).item()


def norm(vec, p=2):
    """Lp norm of a torch vector, returned as a python float."""
    with torch.no_grad():
        return torch.linalg.norm(vec, ord=p).item()
