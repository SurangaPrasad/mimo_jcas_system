import torch.nn as nn
import torch.optim as optim
import torch
from .support_functions_torch import compute_rate_torch, compute_tau_torch, gradient_R_A_torch, gradient_R_D_torch, gradient_tau_A_torch,  gradient_tau_D_torch

# Projection functions for PyTorch
def project_unit_modulus(A):
    """
    Project each entry of A onto the complex unit circle:
    A_proj[i,j] = exp(1j * angle(A[i,j]))
    """
    return torch.exp(1j * torch.angle(A))


def project_power_constraint(A, D, P_BS):
    """
    Project digital precoder D such that ||A D||_F = sqrt(P_BS) using Frobenius norm.
    Works for batched or single inputs.
    """
    if A.dim() == 2:
        # single sample
        norm_AD = torch.norm(A @ D, p='fro')  # Frobenius norm ||A D||_F
        return D * (torch.sqrt(P_BS) / (norm_AD + 1e-12))
    else:
        # batched case
        AD = torch.bmm(A, D)  # (B, N, K)
        norm_AD = torch.norm(AD, dim=(1, 2), p='fro')  # (B,)
        scale = torch.sqrt(P_BS.view(-1).to(AD.device)) / (norm_AD + 1e-12)  # (B,)
        scale = scale.view(-1, 1, 1)
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
        self.mu = nn.Parameter(torch.full((J,), 0.01, dtype=torch.float32))
        self.lambda_ = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, H, A, D, Psi, sigma_n2, P_BS):
        """
        Supports both single and batched inputs:
        - Single: H (M, N), A (N, M), D (M, K)
        - Batched: H (B, M, N), A (B, N, M), D (B, M, K)
        """
        # J inner updates for analog precoder
        A_hat = A.clone()
        # mu = torch.nn.functional.softplus(self.mu)
        # lambda_ = torch.nn.functional.softplus(self.lambda_)
        
        for j in range(self.J):
            # Compute gradients - now supports batching
            grad_RA = gradient_R_A_torch(H, A_hat, D, sigma_n2)
            grad_tauA = gradient_tau_A_torch(A_hat, D, Psi)

            # Gradient ascent with learnable step size
            A_hat = A_hat + self.mu[j] * (grad_RA - self.omega * grad_tauA)

            # Unit modulus projection
            A_hat = project_unit_modulus(A_hat)

        A = A_hat

        # Digital precoder update
        with torch.no_grad():
            grad_RD = gradient_R_D_torch(H, A, D, sigma_n2)
            grad_tauD = gradient_tau_D_torch(A, D, Psi)

        # Gradient ascent with learnable step size
        D = D + self.lambda_ * (grad_RD - self.omega * self.eta * grad_tauD)
        
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
            H:       (B, M, N)  
            A0:      (B, N, M)
            D0:      (B, M, K)
            Psi:     (B, M, M)
            P_BS:    (B,)
        For single-sample, shapes without leading batch dim are accepted too.
        """
        A, D = A0, D0
        for i in range(self.I_max):
            # The layer now handles both single and batched inputs
            A, D = self.layers[i](H, A, D, Psi, sigma_n2, P_BS)
        return A, D


def upganet_loss(H, A, D, Psi, sigma_n2, omega):
    """
    Compute loss for UPGANet training
    Loss = -(R - ω·τ) where we want to maximize (R - ω·τ)
    """
    R = compute_rate_torch(H, A, D, sigma_n2)
    tau = compute_tau_torch(A, D, Psi)
    loss = -(R - omega * tau)
    loss = loss.mean()
    # print("R:", R, "tau:", tau, "loss:", loss)
    return loss 

