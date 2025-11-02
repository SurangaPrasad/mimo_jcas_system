import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import uuid
import torch
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Wavenumber and antenna spacing
lambda_wave = 1  # Wavelength (normalized)
k = 2 * np.pi / lambda_wave
d = lambda_wave / 2  # Antenna spacing

k_t = torch.tensor(k, dtype=torch.float32, device=device)
d_t = torch.tensor(d, dtype=torch.float32, device=device)

# Step 3: Channel Matrix Generation (Saleh-Valenzuela Model) - PyTorch Version
def generate_channel_torch(N, M, L, device, k_t, d_t):
    """Generate channel using PyTorch tensors"""
    H = torch.zeros((M, N), dtype=torch.cfloat, device=device)
    
    for _ in range(L):
        # Complex gain
        alpha_real = torch.randn(1, device=device) / np.sqrt(2)
        alpha_imag = torch.randn(1, device=device) / np.sqrt(2)
        alpha = torch.complex(alpha_real, alpha_imag).squeeze()
        
        # Angles
        phi_r = torch.rand(1, device=device).item() * 2 * np.pi
        phi_t = torch.rand(1, device=device).item() * 2 * np.pi
        
        # Steering vectors
        a_r_phase = 1j * k_t * d_t * torch.arange(M, dtype=torch.float32, device=device) * np.sin(phi_r)
        a_r = torch.exp(a_r_phase) / np.sqrt(M)
        
        a_t_phase = 1j * k_t * d_t * torch.arange(N, dtype=torch.float32, device=device) * np.sin(phi_t)
        a_t = torch.exp(a_t_phase) / np.sqrt(N)
        
        H += np.sqrt(N * M / L) * alpha * torch.outer(a_r, a_t.conj())
    
    return H

def generate_channel_torch_batch(N, M, L, batch_size, device, k_t=None, d_t=None):
    """Generate a batch of channels for training"""
    # Use global values if not provided
    if k_t is None:
        lambda_wave = 1
        k = 2 * np.pi / lambda_wave
        k_t = torch.tensor(k, dtype=torch.float32, device=device)
    if d_t is None:
        lambda_wave = 1
        d = lambda_wave / 2
        d_t = torch.tensor(d, dtype=torch.float32, device=device)
    
    H_batch = torch.zeros((batch_size, M, N), dtype=torch.cfloat, device=device)
    for b in range(batch_size):
        H_batch[b] = generate_channel_torch(N, M, L, device, k_t, d_t)
    return H_batch

# Steering vector function - PyTorch version
def steering_vector_torch(theta, N, device=device, k_t=k_t , d_t=d_t):
    """Compute steering vector using PyTorch"""
    phase = 1j * k_t * d_t * torch.arange(N, dtype=torch.float32, device=device) * torch.sin(torch.tensor(theta, device=device))
    return torch.exp(phase) / torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))

# Compute communication rate R - PyTorch version with batch support
def compute_rate_torch(H, A, D, sigma_n2):
    """
    Vectorized computation of achievable rate (batch or single).
    - H: (B, K, N) or (K, N)
    - A: (B, N, M) or (N, M)
    - D: (B, M, K) or (M, K)
    - sigma_n2: scalar noise power
    Returns: (B,) or scalar
    """
    # Handle single sample by adding batch dimension
    if H.dim() == 2:
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        single = True
    else:
        single = False

    # (B, K, N) @ (B, N, M) -> (B, K, M)
    H_eff = torch.bmm(H, A)

    # (B, K, M) @ (B, M, K) -> (B, K, K)
    G = torch.bmm(H_eff, D)

    # Power coupling matrix (B, K, K)
    P = torch.abs(G) ** 2

    # Desired signal (diagonal)
    signal = torch.diagonal(P, dim1=-2, dim2=-1)

    # Total received power per user
    total_power = P.sum(dim=-1)

    # Interference = total - desired
    interference = total_power - signal + sigma_n2

    rate = torch.log2(1 + signal / interference).sum(dim=-1)

    return rate.squeeze(0) if single else rate



def compute_tau_torch(A, D, Psi):
    """
    Compute sensing error τ = || A D Dᴴ Aᴴ - Ψ ||_F²
    Fully vectorized for both single and batched inputs.
    
    Parameters
    ----------
    A : (N, M) or (B, N, M)
    D : (M, K) or (B, M, K)
    Psi : (M, M) or (B, M, M)
    
    Returns
    -------
    tau : scalar or (B,)
    """
    # --- Ensure all are at least 3D for uniform math ---
    if A.dim() == 2:
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        Psi = Psi.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Compute U = A D Dᴴ Aᴴ   → (B, N, N)
    # Step 1: D Dᴴ
    DDH = torch.bmm(D, D.conj().transpose(1, 2))     # (B, M, M)
    # Step 2: A (D Dᴴ)
    ADDH = torch.bmm(A, DDH)                         # (B, N, M)
    # Step 3: (A D Dᴴ) Aᴴ
    U = torch.bmm(ADDH, A.conj().transpose(1, 2))    # (B, N, N)

    # If Psi is same for all batches → broadcast
    if Psi.shape[0] != A.shape[0]:
        Psi = Psi.expand(A.shape[0], -1, -1)

    # --- τ = ||U - Ψ||_F² per batch ---
    diff = U - Psi
    tau = torch.sum(torch.abs(diff)**2, dim=(1, 2))  # (B,)

    # Return scalar if single sample
    if squeeze_output:
        tau = tau.squeeze(0)
    return tau



# Gradients - PyTorch versions with full batch support
def gradient_R_A_torch(H, A, D, sigma_n2=1, eps=1e-10):
    """
    Compute gradient of R w.r.t. A using PyTorch with numerical stability.
    Supports both single and batched inputs:
    - Single: H (K, N), A (N, M), D (M, K)
    - Batched: H (B, K, N), A (B, N, M), D (B, M, K)
    """
    xi = 1 / torch.log(torch.tensor(2.0, device=A.device))
    
    # Check if batched
    is_batched = (A.dim() == 3)
    
    if is_batched:
        B, N, M = A.shape
        K = D.shape[-1]
        grad_A = torch.zeros_like(A, dtype=torch.cfloat)
        
        # V = D @ D^H for each batch: (B, M, M)
        V = torch.bmm(D, D.conj().transpose(1, 2))
        
        for k in range(K):
            # h_k: (B, N, 1)
            h_k = H[:, k, :].unsqueeze(-1)
            # H_tilde_k: (B, N, N)
            H_tilde_k = torch.bmm(h_k, h_k.conj().transpose(1, 2))
            
            # D_bar_k: D with k-th column zeroed
            D_bar_k = D.clone()
            D_bar_k[:, :, k] = 0.0
            V_bar_k = torch.bmm(D_bar_k, D_bar_k.conj().transpose(1, 2))
            
            # Compute denominators using batch trace
            # trace(A @ V @ A^H @ H_tilde_k) for each batch
            AV = torch.bmm(A, V)  # (B, N, M)
            AVA = torch.bmm(AV, A.conj().transpose(1, 2))  # (B, N, N)
            AVAH = torch.bmm(AVA, H_tilde_k)  # (B, N, N)
            denom1 = torch.diagonal(AVAH, dim1=-2, dim2=-1).sum(dim=-1) + sigma_n2 + eps  # (B,)
            
            AV_bar = torch.bmm(A, V_bar_k)
            AVA_bar = torch.bmm(AV_bar, A.conj().transpose(1, 2))
            AVAH_bar = torch.bmm(AVA_bar, H_tilde_k)
            denom2 = torch.diagonal(AVAH_bar, dim1=-2, dim2=-1).sum(dim=-1) + sigma_n2 + eps  # (B,)
            
            # Compute terms: (B, N, M)
            term1 = torch.bmm(H_tilde_k, torch.bmm(A, V)) / denom1.view(B, 1, 1)
            term2 = torch.bmm(H_tilde_k, torch.bmm(A, V_bar_k)) / denom2.view(B, 1, 1)
            
            grad_A += xi * (term1 - term2)
    else:
        # Single sample case
        K = H.shape[0]
        grad_A = torch.zeros_like(A, dtype=torch.cfloat)
        V = D @ D.conj().T
        
        for k in range(K):
            h_k = H[k, :].reshape(-1, 1)
            H_tilde_k = h_k @ h_k.conj().T
            
            D_bar_k = D.clone()
            D_bar_k[:, k] = 0.0
            V_bar_k = D_bar_k @ D_bar_k.conj().T
            
            denom1 = torch.trace(A @ V @ A.conj().T @ H_tilde_k) + sigma_n2 + eps
            denom2 = torch.trace(A @ V_bar_k @ A.conj().T @ H_tilde_k) + sigma_n2 + eps
            
            term1 = H_tilde_k @ A @ V / denom1
            term2 = H_tilde_k @ A @ V_bar_k / denom2
            
            grad_A += xi * (term1 - term2)
    
    return grad_A


def gradient_R_D_torch(H, A, D, sigma_n2, eps=1e-10, clip_value=1e3):
    """
    Compute gradient of R w.r.t. D using PyTorch with numerical stability.
    Supports both single and batched inputs:
    - Single: H (K, N), A (N, M), D (M, K)
    - Batched: H (B, K, N), A (B, N, M), D (B, M, K)
    """
    xi = 1 / torch.log(torch.tensor(2.0, device=D.device))
    
    # Check if batched
    is_batched = (D.dim() == 3)
    
    if is_batched:
        B, M, K = D.shape
        grad_D = torch.zeros_like(D, dtype=torch.cfloat)
        
        for k in range(K):
            # h_k: (B, N, 1)
            h_k = H[:, k, :].unsqueeze(-1)
            # H_tilde_k: (B, N, N)
            H_tilde_k = torch.bmm(h_k, h_k.conj().transpose(1, 2))
            
            # H_bar_k = A^H @ H_tilde_k @ A: (B, M, M)
            H_bar_k = torch.bmm(torch.bmm(A.conj().transpose(1, 2), H_tilde_k), A)
            
            # D_bar_k: D with k-th column zeroed
            D_bar_k = D.clone()
            D_bar_k[:, :, k] = 0.0
            
            # Compute denominators using batch trace
            DD = torch.bmm(D, D.conj().transpose(1, 2))  # (B, M, M)
            DDH = torch.bmm(DD, H_bar_k)  # (B, M, M)
            denom1 = torch.diagonal(DDH, dim1=-2, dim2=-1).sum(dim=-1) + sigma_n2 + eps  # (B,)
            
            DD_bar = torch.bmm(D_bar_k, D_bar_k.conj().transpose(1, 2))
            DDH_bar = torch.bmm(DD_bar, H_bar_k)
            denom2 = torch.diagonal(DDH_bar, dim1=-2, dim2=-1).sum(dim=-1) + sigma_n2 + eps  # (B,)
            
            # Compute terms: (B, M, K)
            term1 = torch.bmm(H_bar_k, D) / denom1.view(B, 1, 1)
            term2 = torch.bmm(H_bar_k, D_bar_k) / denom2.view(B, 1, 1)
            
            grad_D += xi * (term1 - term2)
        
        # Clip gradients per batch
        grad_norms = torch.linalg.norm(grad_D, ord='fro', dim=(1, 2))  # (B,)
        scale = torch.minimum(torch.ones_like(grad_norms), clip_value / (grad_norms + eps))
        grad_D = grad_D * scale.view(B, 1, 1)
    else:
        # Single sample case
        K = H.shape[0]
        grad_D = torch.zeros_like(D, dtype=torch.cfloat)
        
        for k in range(K):
            h_k = H[k, :].reshape(-1, 1)
            H_tilde_k = h_k @ h_k.conj().T
            H_bar_k = A.conj().T @ H_tilde_k @ A
            
            D_bar_k = D.clone()
            D_bar_k[:, k] = 0.0
            
            denom1 = torch.trace(D @ D.conj().T @ H_bar_k) + sigma_n2 + eps
            denom2 = torch.trace(D_bar_k @ D_bar_k.conj().T @ H_bar_k) + sigma_n2 + eps
            
            term1 = (H_bar_k @ D) / denom1
            term2 = (H_bar_k @ D_bar_k) / denom2
            
            grad_D += xi * (term1 - term2)
        
        # Clip gradients
        grad_norm = torch.linalg.norm(grad_D, ord='fro')
        if grad_norm > clip_value:
            grad_D = grad_D * (clip_value / grad_norm)
    
    return grad_D


def gradient_tau_A_torch(A, D, Psi, eps=1e-10):
    """
    Compute gradient of tau w.r.t. A.
    Supports both single and batched inputs:
    - Single: A (N, M), D (M, K), Psi (M, M)
    - Batched: A (B, N, M), D (B, M, K), Psi (B, M, M)
    """
    is_batched = (A.dim() == 3)
    
    if is_batched:
        B = A.shape[0]
        # U = A @ D @ D^H @ A^H: (B, N, N)
        DD = torch.bmm(D, D.conj().transpose(1, 2))  # (B, M, M)
        ADD = torch.bmm(A, DD)  # (B, N, M)
        U = torch.bmm(ADD, A.conj().transpose(1, 2))  # (B, N, N)
        
        # grad_A = 2 * (U - Psi) @ A @ D @ D^H: (B, N, M)
        U_minus_Psi = U - Psi.view(B, -1, Psi.shape[-1]) if Psi.dim() == 2 else U - Psi
        grad_A = 2 * torch.bmm(torch.bmm(U_minus_Psi, A), DD)
        
        # Normalize per batch
        grad_norms = torch.linalg.norm(grad_A, ord='fro', dim=(1, 2), keepdim=True)  # (B, 1, 1)
        grad_A = grad_A / (grad_norms + eps)
    else:
        # Single sample case
        U = A @ D @ D.conj().T @ A.conj().T
        grad_A = 2 * (U - Psi) @ A @ D @ D.conj().T
        grad_A = grad_A / (torch.linalg.norm(grad_A, ord='fro') + eps)
    
    return grad_A


def gradient_tau_D_torch(A, D, Psi, eps=1e-10):
    """
    Compute gradient of tau w.r.t. D.
    Supports both single and batched inputs:
    - Single: A (N, M), D (M, K), Psi (M, M)
    - Batched: A (B, N, M), D (B, M, K), Psi (B, M, M)
    """
    is_batched = (D.dim() == 3)
    
    if is_batched:
        B = D.shape[0]
        # U = A @ D @ D^H @ A^H: (B, N, N)
        DD = torch.bmm(D, D.conj().transpose(1, 2))  # (B, M, M)
        ADD = torch.bmm(A, DD)  # (B, N, M)
        U = torch.bmm(ADD, A.conj().transpose(1, 2))  # (B, N, N)
        
        # grad_D = 2 * A^H @ (U - Psi) @ A @ D: (B, M, K)
        U_minus_Psi = U - Psi.view(B, -1, Psi.shape[-1]) if Psi.dim() == 2 else U - Psi
        grad_D = 2 * torch.bmm(torch.bmm(A.conj().transpose(1, 2), U_minus_Psi), torch.bmm(A, D))
        
        # Normalize per batch
        grad_norms = torch.linalg.norm(grad_D, ord='fro', dim=(1, 2), keepdim=True)  # (B, 1, 1)
        grad_D = grad_D / (grad_norms + eps)
    else:
        # Single sample case
        U = A @ D @ D.conj().T @ A.conj().T
        grad_D = 2 * A.conj().T @ (U - Psi) @ A @ D
        grad_D = grad_D / (torch.linalg.norm(grad_D, ord='fro') + eps)
    
    return grad_D



def proposed_initialization_torch(H, theta_d, N, M, K, P_BS, device):
    """Proposed initialization using PyTorch"""
    G = H.T  # since M=K
    A0 = torch.exp(-1j * torch.angle(G))[:, :M]
    X_ZF = torch.linalg.pinv(H)
    D0 = torch.linalg.pinv(A0) @ X_ZF
    D0 = torch.sqrt(P_BS) * D0 / torch.linalg.norm(A0 @ D0, ord='fro')
    return A0, D0

def proposed_initialization_torch_batch(H, theta_d, N, M, K, P_BS, device):
    """
    Proposed initialization using PyTorch (batch version)
    H: (batch_size, M, N)
    Returns:
        A0: (batch_size, N, M)
        D0: (batch_size, M, N)
    """
    batch_size = H.shape[0]

    # G = H^T for each batch (transpose last two dims)
    G = H.transpose(-1, -2)  # shape: (batch_size, N, M)
    
    # A0 = exp(-j * angle(G))[:, :M]
    A0 = torch.exp(-1j * torch.angle(G))[:, :, :M]  # shape: (batch_size, N, M)
    
    # Compute pseudo-inverse for each batch element
    X_ZF = torch.linalg.pinv(H)  # shape: (batch_size, N, M)
    
    # D0 = pinv(A0) @ X_ZF for each batch
    A0_pinv = torch.linalg.pinv(A0)  # shape: (batch_size, M, N)
    D0 = torch.bmm(A0_pinv, X_ZF)  # shape: (batch_size, M, M)
    
    # Normalize
    norm_factor = torch.linalg.norm(torch.bmm(A0, D0), ord='fro', dim=(1, 2), keepdim=True)
    D0 = torch.sqrt(P_BS) * D0 / norm_factor

    return A0, D0

import torch

def proposed_initialization_torch_batch_multiSNR(H, theta_d, N, M, K, P_BS_list, device):
    """
    Proposed initialization using PyTorch (batch version with multi-SNR support)

    Args:
        H: (batch_size, M, N) complex tensor (channels)
        theta_d: any design param (unused here, but kept for interface consistency)
        N, M, K: system dimensions
        P_BS_list: list or tensor of possible P_BS values (one per SNR)
        device: torch device

    Returns:
        A0: (batch_size, N, M) complex tensor (analog precoder)
        D0: (batch_size, M, N) complex tensor (digital precoder)
        P_BS_used: (batch_size,) float tensor (which P_BS used for each sample)
    """

    batch_size = H.shape[0]
    P_BS_list = torch.as_tensor(P_BS_list, dtype=torch.float32, device=device)

    # === Randomly sample one P_BS per batch sample ===
    rand_indices = torch.randint(0, len(P_BS_list), (batch_size,), device=device)
    P_BS_used = P_BS_list[rand_indices]  # (batch_size,)

    # === Compute G, A0 ===
    G = H.transpose(-1, -2).clone()  # (batch_size, N, M)
    A0 = torch.exp(-1j * torch.angle(G))[:, :, :M]  # (batch_size, N, M)

    # === Compute pseudo-inverses ===
    # torch.linalg.pinv supports batched input
    X_ZF = torch.linalg.pinv(H)              # (batch_size, N, M)
    A0_pinv = torch.linalg.pinv(A0)          # (batch_size, M, N)
    D0 = torch.bmm(A0_pinv, X_ZF)            # (batch_size, M, M)

    # === Normalize to satisfy power constraint ===
    # Compute ||A0 @ D0||_F for each batch
    norm_factor = torch.linalg.norm(torch.bmm(A0, D0), ord='fro', dim=(1, 2), keepdim=True)  # (batch_size, 1, 1)
    
    # Expand P_BS_used to match D0 shape for broadcasting
    P_BS_expand = torch.sqrt(P_BS_used).view(batch_size, 1, 1)  # (batch_size, 1, 1)
    
    D0 = D0 * (P_BS_expand / norm_factor)

    return A0, D0, P_BS_used


def random_initialization_torch(N, M, H, P_BS, device):
    """Random initialization using PyTorch"""
    A0 = torch.exp(1j * torch.rand((N, M), device=device) * 2 * np.pi)
    H_A = H @ A0
    D0 = torch.linalg.pinv(H_A)
    D0 = torch.sqrt(torch.tensor(P_BS, device=device)) * D0 / torch.linalg.norm(A0 @ D0, ord='fro')
    return A0, D0

def svd_initialization_torch(H, N, M, K, P_BS, device):
    """SVD initialization using PyTorch"""
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    A0 = Vh.T[:, :M]  # Take first M columns
    A0 = torch.exp(1j * torch.angle(A0))  # Project to unit modulus
    H_A = H @ A0
    try:
        D0 = torch.linalg.pinv(H_A)
    except:
        D0 = torch.linalg.pinv(H_A + 1e-6 * torch.eye(M, device=device, dtype=torch.cfloat))
    D0 = torch.sqrt(torch.tensor(P_BS, device=device)) * D0 / torch.linalg.norm(A0 @ D0, ord='fro')
    return A0, D0


# PGA Algorithm - PyTorch Version
def run_pga_torch(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, P_BS, device):
    """
    Run PGA algorithm using PyTorch tensors.
    All inputs should be PyTorch tensors on the specified device.
    Returns the convergence history of R - omega * tau at each iteration.
    """
    N, K = H.shape[1], H.shape[0]
    A = A0.clone()
    D = D0.clone()
    
    eta = 1.0 / N  # Balancing term
    
    # Track objective value at each iteration
    objective_history = np.zeros(I_max)

    for i in range(I_max):
        # ---- Inner Loop: Analog Precoder Update ----
        A_hat = A.clone()
        
        for j in range(J):
            grad_R_A = gradient_R_A_torch(H, A_hat, D, sigma_n2)
            grad_tau_A = gradient_tau_A_torch(A_hat, D, Psi)

            # Eq. (14b): Gradient Ascent on A
            grad_A = grad_R_A - omega * grad_tau_A
            A_hat = A_hat + mu * grad_A

            # Eq. (7): Unit Modulus Projection
            A_hat = torch.exp(1j * torch.angle(A_hat))

        A = A_hat.clone()

        # ---- Outer Loop: Digital Precoder Update ----
        grad_R_D = gradient_R_D_torch(H, A, D, sigma_n2)
        grad_tau_D = gradient_tau_D_torch(A, D, Psi)

        # Eq. (15): Gradient Ascent on D
        grad_D = grad_R_D - omega * eta * grad_tau_D
        D = D + lambda_ * grad_D

        # Eq. (9): Power Constraint Projection
        D = torch.sqrt(P_BS) * D / torch.linalg.norm(A @ D, ord='fro')
        
        # Compute objective value: R - omega * tau
        R = compute_rate_torch(H, A, D, sigma_n2)
        tau = compute_tau_torch(A, D, Psi)
        objective_history[i] = (R - omega * tau).cpu().item()

    return objective_history, R, A, D


# Zero-Forcing Baseline - PyTorch Version (FIXED)
def compute_R_ZF_torch(H, sigma_n2, P_BS, device=device):
    """
    Compute the achievable sum rate using Zero-Forcing precoding with PyTorch.
    H: (K x N) channel matrix (K users, N antennas)
    """
    K, N = H.shape

    # Ensure correct orientation (K x N)
    if H.shape[0] != K:
        H = H.T

    # ZF precoder
    X_ZF = H.conj().T @ torch.linalg.pinv(H @ H.conj().T)  # (N x K)

    # Normalize total transmit power
    X_ZF = torch.sqrt(P_BS) * X_ZF / torch.linalg.norm(X_ZF, ord='fro')

    # Compute sum rate
    R_ZF = torch.tensor(0.0, device=device)
    for k in range(K):
        h_k = H[k, :].reshape(1, -1)  # (1 x N)
        signal = torch.abs(h_k @ X_ZF[:, k])**2
        signal = signal.squeeze()  # Remove dimensions of size 1
        
        interference = torch.sum(torch.abs(h_k @ X_ZF)**2) - signal
        interference = interference.squeeze()  # Remove dimensions of size 1
        
        SINR = signal / (interference + sigma_n2)
        R_ZF = R_ZF + torch.log2(1 + SINR)
    
    return torch.real(R_ZF)

SNR_dB_array = np.arange(0, 12.1, 0.1)
# Load Psi data (from MATLAB .mat file)
with h5py.File('Psi_all.mat', 'r') as f:
    Psi_h5 = f['Psi_all']

    # If stored as MATLAB complex structure (real/imag parts separate)
    if np.issubdtype(Psi_h5.dtype, np.void):
        real = Psi_h5['real'][()]
        imag = Psi_h5['imag'][()]
        Psi_all = real + 1j * imag
    else:
        Psi_all = np.array(Psi_h5)

# Ensure Psi_all has shape: (num_SNRs, M, N)
Psi_all = np.squeeze(Psi_all)  # remove singleton dimensions if any


def compute_psi(snr_db):
    """
    Selects the Psi matrix corresponding to the closest SNR value.
    """
    # Find index of closest SNR
    idx = np.argmin(np.abs(SNR_dB_array - snr_db))

    # Select corresponding Psi
    Psi = Psi_all[idx, :, :]

    return Psi

def evaluate_trained_upganet(N, M, K,  model_path, theta_d_t, sigma_n2_t, omega, sigma_n2, I_max,  J, num_realizations=100, snr_range=range(0, 13 , 2)):
    """
    Evaluate a trained UPGANet model and compute the average achievable rate vs SNR.
    """
    # Import here to avoid circular import
    from .dnn import UPGANet
    
    print("="*70)
    print(f"EVALUATING TRAINED UPGANet (J={J})")
    print("="*70)
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print()

    # Instantiate model with the same I_max as training
    trained_model = UPGANet(N, M, K, omega, I_max=I_max, J=J).to(device)
    state_dict = torch.load(model_path, map_location=device)
    trained_model.load_state_dict(state_dict, strict=False)
    trained_model.eval()

    R_upganet_list = []

    for snr_db in snr_range:
        R_sum = 0.0
        P_BS = sigma_n2 * 10**(snr_db / 10)
        P_BS_t = torch.tensor(P_BS, dtype=torch.float32, device=device)
        H_batch = generate_channel_torch_batch(N, M, L=20, batch_size=num_realizations, device=device)
        Psi_t = torch.tensor(compute_psi(snr_db), dtype=torch.cfloat, device=device)

        for realization in range(num_realizations):
            H_t = H_batch[realization]
            # Initialization (same as before)
            A0, D0 = proposed_initialization_torch(H_t, theta_d_t, N, M, K, P_BS_t, device=device)

            with torch.no_grad():
                A_final, D_final = trained_model(H_t, A0, D0, Psi_t, sigma_n2_t, P_BS_t)

            # Compute achievable rate
            R = compute_rate_torch(H_t, A_final, D_final, sigma_n2_t)
            R_sum += R.item()

        R_avg = R_sum / num_realizations
        R_upganet_list.append(R_avg)
        print(f"SNR = {snr_db:2d} dB → Avg R = {R_avg:.4f} bps/Hz")

    return R_upganet_list
