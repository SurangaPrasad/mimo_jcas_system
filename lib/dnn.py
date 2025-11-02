import torch.nn as nn
import torch.optim as optim
import torch
from .support_functions_torch import compute_rate_torch, compute_tau_torch, gradient_R_A_torch, gradient_R_D_torch, gradient_tau_A_torch,  gradient_tau_D_torch

# Projection functions for PyTorch
def project_unit_modulus(A):
    """Project A onto unit modulus constraint"""
    return A / (torch.abs(A) + 1e-8)

def project_power_constraint(A, D, P_BS):
    """
    A: (B, N, M) or (N, M) if single sample
    D: (B, M, K) or (M, K)
    P_BS: (B,) or scalar
    Returns scaled D, same device/dtype.
    """
    if A.dim() == 2:
        # single sample
        norm = torch.linalg.norm(A @ D, ord='fro')
        return D * (torch.sqrt(P_BS) / (norm + 1e-8))
    else:
        # batched
        # compute A @ D for each sample: (B, N, K)
        AD = torch.bmm(A, D)                    # (B, N, K)
        norms = torch.linalg.norm(AD, dim=(1,2))  # (B,)
        scale = torch.sqrt(P_BS).view(-1).to(norms.device) / (norms + 1e-8)  # (B,)
        scale = scale.view(-1, 1, 1)            # (B,1,1)
        return D * scale


class UPGANetLayer(nn.Module):
    """Single layer of UPGANet with learnable step sizes"""
    def __init__(self, N, M, K, omega, J=10, eta=None):
        super(UPGANetLayer, self).__init__()
        self.J = J
        self.N, self.M, self.K = N, M, K
        self.omega = omega
        self.eta = eta if eta is not None else 1.0 / N

        # Learnable step sizes (one for each inner iteration)
        self.mu = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, H, A, D, Psi, sigma_n2, P_BS):

        # J inner updates for analog precoder
        A_hat = A.clone()
        mu = torch.nn.functional.softplus(self.mu)
        lambda_ = torch.nn.functional.softplus(self.lambda_)
        for j in range(self.J):

            # Compute gradients using PyTorch functions
            # DETACH to prevent computing gradients of gradients!
            # with torch.no_grad():
            grad_RA = gradient_R_A_torch(H, A_hat, D, sigma_n2)
            grad_tauA = gradient_tau_A_torch(A_hat, D, Psi)

            # Gradient ascent with learnable step size
            # The step sizes (mu) remain in the computational graph for learning
            A_hat = A_hat + mu * (grad_RA - self.omega * grad_tauA)

            # Unit modulus projection
            A_hat = project_unit_modulus(A_hat)

        A = A_hat

        # Digital precoder update
        # DETACH gradient computations here too
        with torch.no_grad():
            grad_RD = gradient_R_D_torch(H, A, D, sigma_n2)
            grad_tauD = gradient_tau_D_torch(A, D, Psi)

        # Gradient ascent with learnable step size
        D = D + lambda_ * (grad_RD - self.omega * self.eta * grad_tauD)
        
        # Power constraint projection
        D = project_power_constraint(A, D, P_BS)

        return A, D


class UPGANet(nn.Module):
    def __init__(self, N, M, K, omega, I_max=120, J=10):
        super().__init__()
        self.N = N; self.M = M; self.K = K; self.omega = omega; self.I_max = I_max
        self.layers = nn.ModuleList([UPGANetLayer(N, M, K, omega, J=J) for _ in range(I_max)])

    def forward(self, H, A0, D0, Psi, sigma_n2, P_BS):
        """
        Support both single-sample and batched inputs.
        Expected shapes (batched):
            H:       (B, K, N) or (B, K, N, M) depending on your H shape convention
            A0:      (B, N, M)
            D0:      (B, M, K)
            Psi:     (B, M, M)
            P_BS:    (B,)
        For single-sample, shapes without leading batch dim are accepted too.
        """
        # Detect batched vs single
        batched = (A0.dim() == 3)  # (B, N, M) -> batched
        if not batched:
            A, D = A0, D0
            for i in range(self.I_max):
                A, D = self.layers[i](H, A, D, Psi, sigma_n2, P_BS)
            return A, D

        # Batched path
        B = A0.shape[0]
        A = A0
        D = D0

        for i in range(self.I_max):
            A_list = []
            D_list = []
            for b in range(B):
                # call per-sample layer (H[b] shape must match layer expectation)
                A_b, D_b = self.layers[i](H[b], A[b], D[b], Psi[b], sigma_n2, P_BS[b])
                A_list.append(A_b)
                D_list.append(D_b)
            A = torch.stack(A_list, dim=0)
            D = torch.stack(D_list, dim=0)

        return A, D


def upganet_loss(H, A, D, Psi, sigma_n2, omega):
    # H, A, D, Psi may be batched
    if A.dim() == 3:  # batched
        B = A.shape[0]
        losses = []
        for b in range(B):
            R = compute_rate_torch(H[b], A[b], D[b], sigma_n2)
            tau = compute_tau_torch(A[b], D[b], Psi[b])
            losses.append(-(R - omega * tau))
        loss = torch.stack([l if isinstance(l, torch.Tensor) else torch.tensor(l, device=A.device) for l in losses]).mean()
    else:
        R = compute_rate_torch(H, A, D, sigma_n2)
        tau = compute_tau_torch(A, D, Psi)
        loss = -(R - omega * tau)
        loss = loss.mean() if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=A.device)
    return loss


