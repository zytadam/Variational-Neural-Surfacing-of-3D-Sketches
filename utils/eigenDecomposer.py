import numpy as np
import scipy as sp
import torch

class EigenDecomposer:
    def __init__(self, device="cpu"):
        self.device = device

    def decompose2D(self, A):
        # input (..., 2, 2)

        trA = A.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        detA = torch.linalg.det(A)
        gapA = torch.sqrt(trA ** 2 - 4 * detA)

        lambda1 = (trA + gapA) / 2
        lambda2 = (trA - gapA) / 2

        I = torch.eye(2, device=self.device, requires_grad=False)
        V1 = A - lambda2[..., None, None] * I
        # V2 = A - lambda1[..., None, None] * I
        
        i1 = torch.argmax(torch.linalg.norm(V1, ord=2, dim=-2), dim=-1).unsqueeze(-1).unsqueeze(-1).expand(*V1.shape[:-1], 1)
        # i2 = torch.argmax(torch.linalg.norm(V2, ord=2, dim=-2), dim=-1).unsqueeze(-1).unsqueeze(-1).expand(*V2.shape[:-1], 1)
        v1 = torch.gather(V1, dim=-1, index=i1).squeeze(-1)
        # v2 = torch.gather(V2, dim=-1, index=i2)

        v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)
        print(v1)
        v2 = torch.stack([-v1[..., 1], v1[..., 0]], dim=-1)
        print(v2)
        return torch.stack([lambda1, lambda2], dim=-1), torch.stack([v1, v2], dim=-1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decomposer = EigenDecomposer(device=device)
    A = torch.randn(1, 1, 2, 2).to(device)
    val, vec = decomposer.decompose2D(0.5 * (A + A.transpose(-2, -1)))

    print(val)
    print(vec)

    val, vec = torch.linalg.eigh(0.5 * (A + A.transpose(-2, -1)))
    print(val)
    print(vec)