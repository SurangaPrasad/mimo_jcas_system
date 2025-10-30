import torch.nn as nn
import torch.optim as optim
import torch
from .support_functions_torch import compute_rate_torch, compute_tau_torch, gradient_R_A_torch, gradient_R_D_torch, gradient_tau_A_torch,  gradient_tau_D_torch

# Projection functions for PyTorch
def project_unit_modulus(A):
    """Project A onto unit modulus constraint"""
    return torch.exp(1j * torch.angle(A))

def project_power_constraint(A, D, P_BS):
    """Project D to satisfy power constraint"""
    norm_factor = torch.sqrt(P_BS) / torch.linalg.norm(A @ D, ord='fro')
    return D * norm_factor

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

        # J inner updates for analog precoder
        A_hat = A.clone()
        for j in range(self.J):
            # Compute gradients using PyTorch functions
            # DETACH to prevent computing gradients of gradients!
            with torch.no_grad():
                grad_RA = gradient_R_A_torch(H, A_hat, D, sigma_n2)
                grad_tauA = gradient_tau_A_torch(A_hat, D, Psi)

            # Gradient ascent with learnable step size
            # The step sizes (mu) remain in the computational graph for learning
            A_hat = A_hat + self.mu[j] * (grad_RA - self.omega * grad_tauA)
            
            # Unit modulus projection
            A_hat = project_unit_modulus(A_hat)

        A = A_hat

        # Digital precoder update
        # DETACH gradient computations here too
        with torch.no_grad():
            grad_RD = gradient_R_D_torch(H, A, D, sigma_n2)
            grad_tauD = gradient_tau_D_torch(A, D, Psi)

        # Gradient ascent with learnable step size
        D = D + self.lambda_ * (grad_RD - self.omega * self.eta * grad_tauD)
        
        # Power constraint projection
        D = project_power_constraint(A, D, P_BS)

        return A, D


class UPGANet(nn.Module):
    """Unfolded Projected Gradient Ascent Network"""
    def __init__(self, N, M, K, omega, I_max=120, J=10):
        super(UPGANet, self).__init__()
        self.N = N
        self.M = M
        self.K = K
        self.omega = omega
        self.I_max = I_max
        
        # Create I_max layers (outer iterations)
        self.layers = nn.ModuleList([
            UPGANetLayer(N, M, K, omega, J=J) for _ in range(I_max)
        ])
    
    def forward(self, H, A0, D0, Psi, sigma_n2, P_BS):

        A, D = A0, D0
        for i in range(self.I_max):
            A, D = self.layers[i](H, A, D, Psi, sigma_n2, P_BS)
        return A, D

def upganet_loss(H, A, D, Psi, sigma_n2, omega):
    """
    Compute loss for UPGANet training
    Loss = -(R - ω·τ) where we want to maximize (R - ω·τ)
    """
    R = compute_rate_torch(H, A, D, sigma_n2)
    tau = compute_tau_torch(A, D, Psi)
    return -(R - omega * tau)  # Negative because we minimize loss

