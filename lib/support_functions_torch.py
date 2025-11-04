import numpy as np
import torch
import h5py

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Channel generation (single + batch) ----
def generate_channel_torch(N, M, L, device=device, k_t=None, d_t=None):
    """Single-sample Saleh-Valenzuela channel (returns H: (K=M, N))"""
    if k_t is None:
        k = 2 * np.pi
        k_t = torch.tensor(k, dtype=torch.float32, device=device)
    if d_t is None:
        d = 0.5
        d_t = torch.tensor(d, dtype=torch.float32, device=device)

    H = torch.zeros((M, N), dtype=torch.cfloat, device=device)
    for _ in range(L):
        alpha = torch.complex(torch.randn(1, device=device)/np.sqrt(2),
                              torch.randn(1, device=device)/np.sqrt(2)).squeeze()
        phi_r = float(torch.rand(1, device=device).item() * 2 * np.pi)
        phi_t = float(torch.rand(1, device=device).item() * 2 * np.pi)

        a_r_phase = 1j * k_t * d_t * torch.arange(M, dtype=torch.float32, device=device) * np.sin(phi_r)
        a_r = torch.exp(a_r_phase) / np.sqrt(M)

        a_t_phase = 1j * k_t * d_t * torch.arange(N, dtype=torch.float32, device=device) * np.sin(phi_t)
        a_t = torch.exp(a_t_phase) / np.sqrt(N)

        H += np.sqrt(N * M / L) * alpha * torch.outer(a_r, a_t.conj())

    return H


def generate_channel_torch_batch(N, M, L, batch_size, device=device, k_t=None, d_t=None):
    """Batch version: returns H_batch shape (B, M, N)"""
    H_batch = torch.zeros((batch_size, M, N), dtype=torch.cfloat, device=device)
    for b in range(batch_size):
        H_batch[b] = generate_channel_torch(N, M, L, device=device, k_t=k_t, d_t=d_t)
    return H_batch


# ---- Steering vector (supports scalar or batched theta) ----
def steering_vector_torch(theta, N, device=device, k_t=None, d_t=None):
    """
    theta: scalar or 1D tensor of angles
    Returns: (N,) or (len(theta), N) complex steering vectors
    """
    if k_t is None:
        k_t = torch.tensor(2 * np.pi, dtype=torch.float32, device=device)
    if d_t is None:
        d_t = torch.tensor(0.5, dtype=torch.float32, device=device)

    if torch.is_tensor(theta) and theta.dim() > 0:
        # batched angles -> return (B, N)
        theta = theta.to(device)
        n = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)  # (1, N)
        phase = 1j * k_t * d_t * n * torch.sin(theta.unsqueeze(1))            # (B, N)
        return torch.exp(phase) / torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))
    else:
        th = torch.tensor(float(theta), device=device)
        n = torch.arange(N, dtype=torch.float32, device=device)
        phase = 1j * k_t * d_t * n * torch.sin(th)
        return torch.exp(phase) / torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))


# ---- compute_rate (vectorized, single+batch) ----
def compute_rate_torch(H, A, D, sigma_n2):
    """
    Vectorized computation of achievable rate (batch or single).
    - H: (B, K, N) or (K, N)
    - A: (B, N, M) or (N, M)
    - D: (B, M, K) or (M, K)
    - sigma_n2: scalar or (B,)
    Returns: (B,) or scalar
    """
    eps = 1e-10
    single = False
    if H.dim() == 2:
        single = True
        H = H.unsqueeze(0)   # (1,K,N)
        A = A.unsqueeze(0)   # (1,N,M)
        D = D.unsqueeze(0)   # (1,M,K)

    B, K, N = H.shape
    # H^H * A  => use H.conj() @ A to obtain (B, K, M)
    H_eff = torch.bmm(H.conj(), A)     # (B, K, M)
    G = torch.bmm(H_eff, D)            # (B, K, K)
    P = torch.abs(G)**2                # (B, K, K) real
    signal = torch.diagonal(P, dim1=-2, dim2=-1)   # (B, K)
    total_power = P.sum(dim=-1)                    # (B, K)
    interference = total_power - signal            # (B, K)

    # sigma_n2 -> broadcast
    if not torch.is_tensor(sigma_n2):
        sigma_n2_t = torch.tensor(float(sigma_n2), dtype=signal.dtype, device=signal.device)
    else:
        sigma_n2_t = sigma_n2.to(signal.device)
    # If sigma_n2 is per-sample (B,), expand
    if sigma_n2_t.dim() == 0:
        sigma_b = sigma_n2_t.view(1)
    else:
        sigma_b = sigma_n2_t.view(-1)

    # denom: (B,K) -> broadcast sigma if necessary
    denom = interference + sigma_b.view(-1, 1)
    denom = torch.clamp(denom, min=eps)

    rate_per_user = torch.log2(1.0 + signal / denom)   # (B,K)
    rate = rate_per_user.sum(dim=-1)                   # (B,)

    return rate.squeeze(0) if single else rate


# ---- compute_tau (batch aware) ----
def compute_tau_torch(A, D, Psi):
    """
    Compute tau = || A D D^H A^H - Psi ||_F^2
    Supports single and batched inputs:
      - A: (B, N, M) or (N, M)
      - D: (B, M, K) or (M, K)
      - Psi: (B, N, N) or (N, N)
    Returns: (B,) or scalar
    """
    single = False
    if A.dim() == 2:
        single = True
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        Psi = Psi.unsqueeze(0) if Psi.dim() == 2 else Psi

    B = A.shape[0]
    DDH = torch.bmm(D, D.conj().transpose(1, 2))           # (B, M, M)
    ADDH = torch.bmm(A, DDH)                              # (B, N, M)
    U = torch.bmm(ADDH, A.conj().transpose(1, 2))         # (B, N, N)

    if Psi.shape[0] != B:
        Psi = Psi.expand(B, -1, -1)

    diff = U - Psi
    tau = torch.linalg.norm(diff, ord='fro', dim=(-2, -1))**2   # (B,)
    return tau.squeeze(0) if single else tau


# ---- gradient_R_A (batched) ----
def gradient_R_A_torch(H, A, D, sigma_n2=1.0, eps=1e-10):
    """
    Gradient of R w.r.t. A, supports batched inputs.
    H: (B,K,N) or (K,N)
    A: (B,N,M) or (N,M)
    D: (B,M,K) or (M,K)
    Returns: gradient shape matching A
    """
    xi = 1.0 / torch.log(torch.tensor(2.0, device=A.device))
    single = False
    if H.dim() == 2:
        single = True
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)

    B, K, N = H.shape
    M = A.shape[2]
    grad_A = torch.zeros_like(A, dtype=torch.cfloat)

    # V = D D^H: (B, M, M)
    V = torch.bmm(D, D.conj().transpose(1, 2))

    # Precompute A V and AV A^H
    AV = torch.bmm(A, V)                                 # (B, N, M)
    AVAH = torch.bmm(AV, A.conj().transpose(1, 2))       # (B, N, N)

    for k in range(K):
        h_k = H[:, k, :].unsqueeze(-1)                   # (B, N, 1)
        H_tilde_k = torch.bmm(h_k, h_k.conj().transpose(1, 2))  # (B, N, N)

        # D_bar_k: zero kth column
        D_bar_k = D.clone()
        D_bar_k[:, :, k] = 0.0
        V_bar_k = torch.bmm(D_bar_k, D_bar_k.conj().transpose(1, 2))

        AVH_k = AVAH                                   # (B, N, N) (reuse)

        # denom1 = trace(A V A^H H_tilde_k) + sigma + eps  -> compute by elementwise multiply and sum diag:
        AVAH_H = torch.bmm(AVAH, H_tilde_k)           # (B, N, N)
        denom1 = torch.diagonal(AVAH_H, dim1=-2, dim2=-1).sum(dim=-1) + float(sigma_n2) + eps   # (B,)

        AV_bar = torch.bmm(A, V_bar_k)
        AVAH_bar = torch.bmm(AV_bar, A.conj().transpose(1, 2))
        AVAH_bar_H = torch.bmm(AVAH_bar, H_tilde_k)
        denom2 = torch.diagonal(AVAH_bar_H, dim1=-2, dim2=-1).sum(dim=-1) + float(sigma_n2) + eps  # (B,)

        # Convert to shape (B,1,1) for broadcasting
        denom1b = denom1.view(B, 1, 1)
        denom2b = denom2.view(B, 1, 1)

        term1 = torch.bmm(H_tilde_k, torch.bmm(A, V)) / denom1b
        term2 = torch.bmm(H_tilde_k, torch.bmm(A, V_bar_k)) / denom2b

        grad_A += xi * (term1 - term2)

    return grad_A.squeeze(0) if single else grad_A


# ---- gradient_R_D (batched) ----
def gradient_R_D_torch(H, A, D, sigma_n2=1.0, eps=1e-10, clip_value=None):
    """
    Gradient of R w.r.t. D, supports batched inputs.
    Returns shape like D.
    """
    xi = 1.0 / torch.log(torch.tensor(2.0, device=A.device))
    single = False
    if H.dim() == 2:
        single = True
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)

    B, K, N = H.shape
    M = D.shape[1]
    grad_D = torch.zeros_like(D, dtype=torch.cfloat)

    for k in range(K):
        h_k = H[:, k, :].unsqueeze(-1)                         # (B, N, 1)
        H_tilde_k = torch.bmm(h_k, h_k.conj().transpose(1, 2)) # (B, N, N)
        H_bar_k = torch.bmm(torch.bmm(A.conj().transpose(1, 2), H_tilde_k), A)  # (B, M, M)

        D_bar_k = D.clone()
        D_bar_k[:, :, k] = 0.0

        DD = torch.bmm(D, D.conj().transpose(1, 2))
        DDH = torch.bmm(DD, H_bar_k)
        denom1 = torch.diagonal(DDH, dim1=-2, dim2=-1).sum(dim=-1) + float(sigma_n2) + eps    # (B,)

        DD_bar = torch.bmm(D_bar_k, D_bar_k.conj().transpose(1, 2))
        DDH_bar = torch.bmm(DD_bar, H_bar_k)
        denom2 = torch.diagonal(DDH_bar, dim1=-2, dim2=-1).sum(dim=-1) + float(sigma_n2) + eps  # (B,)

        denom1b = denom1.view(B, 1, 1)
        denom2b = denom2.view(B, 1, 1)

        term1 = torch.bmm(H_bar_k, D) / denom1b
        term2 = torch.bmm(H_bar_k, D_bar_k) / denom2b

        grad_D += xi * (term1 - term2)

    # optional clipping per-sample
    if clip_value is not None:
        norms = torch.linalg.norm(grad_D, ord='fro', dim=(1, 2))
        scale = torch.minimum(torch.ones_like(norms), (clip_value / (norms + eps)))
        grad_D = grad_D * scale.view(B, 1, 1)

    return grad_D.squeeze(0) if single else grad_D


# ---- gradient_tau_A and gradient_tau_D (batched) ----
def gradient_tau_A_torch(A, D, Psi, eps=1e-12):
    """
    grad_A = 2 * (U - Psi) @ A @ (D D^H)
    Supports batched A/D/Psi or single samples.
    """
    single = False
    if A.dim() == 2:
        single = True
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        if Psi.dim() == 2:
            Psi = Psi.unsqueeze(0)

    B = A.shape[0]
    DD = torch.bmm(D, D.conj().transpose(1, 2))
    ADD = torch.bmm(A, DD)
    U = torch.bmm(ADD, A.conj().transpose(1, 2))

    if Psi.shape[0] != B:
        Psi = Psi.expand(B, -1, -1)

    U_minus_Psi = U - Psi
    grad_A = 2 * torch.bmm(torch.bmm(U_minus_Psi, A), DD)

    # normalize per-sample to avoid explosion (optional)
    # norms = torch.linalg.norm(grad_A, ord='fro', dim=(1, 2), keepdim=True)
    # grad_A = grad_A / (norms + eps)

    return grad_A.squeeze(0) if single else grad_A


def gradient_tau_D_torch(A, D, Psi, eps=1e-12, clip_thresh=1e3):
    """
    grad_D = 2 * A^H @ (U - Psi) @ A @ D
    Supports batch inputs.
    """
    single = False
    if A.dim() == 2:
        single = True
        A = A.unsqueeze(0)
        D = D.unsqueeze(0)
        if Psi.dim() == 2:
            Psi = Psi.unsqueeze(0)

    B = A.shape[0]
    DD = torch.bmm(D, D.conj().transpose(1, 2))
    ADD = torch.bmm(A, DD)
    U = torch.bmm(ADD, A.conj().transpose(1, 2))

    if Psi.shape[0] != B:
        Psi = Psi.expand(B, -1, -1)

    U_minus_Psi = U - Psi
    grad_D = 2 * torch.bmm(torch.bmm(A.conj().transpose(1, 2), U_minus_Psi), torch.bmm(A, D))

    # normalize per-sample
    # norms = torch.linalg.norm(grad_D, ord='fro', dim=(1, 2), keepdim=True)
    # grad_D = grad_D / (norms + eps)

    norms = torch.linalg.norm(grad_D, ord='fro', dim=(1, 2), keepdim=True)
    scale = torch.clamp(clip_thresh / (norms + eps), max=1.0)  # only scales down
    grad_D = grad_D * scale

    return grad_D.squeeze(0) if single else grad_D


# ---- Initializations (batched + single) ----
def proposed_initialization_torch(H, theta_d, N, M, K, P_BS, device=device):
    """Single-sample proposed initialization (uses pinned shapes)."""
    single = False
    if H.dim() == 3:
        raise ValueError("Use proposed_initialization_torch_batch for batched input.")
    # H: (K,N)
    eps = 1e-12
    G = H.transpose(0, 1)                       # (N, K)
    A0 = torch.exp(-1j * torch.angle(G))[:, :M] # (N, M)
    X_ZF = torch.linalg.pinv(H)                 # (N, K)
    A0_pinv = torch.linalg.pinv(A0)             # (M, N)
    D0 = A0_pinv @ X_ZF                         # (M, K)
    norm_AD = torch.linalg.norm(A0 @ D0, ord='fro')
    D0 = D0 * (torch.sqrt(torch.tensor(P_BS, dtype=norm_AD.dtype, device=device)) / (norm_AD + eps))
    return A0, D0


def proposed_initialization_torch_batch(H, theta_d, N, M, K, P_BS, device=device):
    """
    H: (B, K, N)
    Returns A0: (B, N, M), D0: (B, M, K)
    """
    eps = 1e-12
    B = H.shape[0]
    G = H.transpose(-1, -2).clone()         # (B, N, K)
    A0 = torch.exp(-1j * torch.angle(G))              # (B, N, M)
    X_ZF = torch.linalg.pinv(H)                     # (B, N, K)
    A0_pinv = torch.linalg.pinv(A0)                 # (B, M, N)
    D0 = torch.bmm(A0_pinv, X_ZF)                   # (B, M, K)
    if not torch.is_tensor(P_BS):
        P_vec = torch.tensor(float(P_BS), device=device).view(1).repeat(B)
    else:
        P_vec = P_BS.view(-1).to(device)
    sqrtP = torch.sqrt(P_vec).view(B, 1, 1)
    norm_AD = torch.linalg.norm(torch.bmm(A0, D0), ord='fro', dim=(1, 2), keepdim=True)
    D0 = D0 * (sqrtP / (norm_AD + eps))
    return A0, D0


def proposed_initialization_torch_batch_multiSNR(H, theta_d, N, M, K, P_BS_list, device=device):
    """
    H: (B, K, N)
    P_BS_list: list or tensor of candidate P_BS values
    Returns: A0 (B,N,M), D0 (B,M,K), P_BS_used (B,)
    """
    B = H.shape[0]
    P_BS_list = torch.as_tensor(P_BS_list, dtype=torch.float32, device=device)
    rand_idx = torch.randint(0, len(P_BS_list), (B,), device=device)
    P_used = P_BS_list[rand_idx]
    A0, D0 = proposed_initialization_torch_batch(H, theta_d, N, M, K, P_used, device=device)
    return A0, D0, P_used


def random_initialization_torch_batch(N, M, H_batch, P_BS, device=device):
    """
    Batch random initialization.
    H_batch: (B, K, N) -> returns A0 (B, N, M), D0 (B, M, K)
    """
    B = H_batch.shape[0]
    eps = 1e-12
    A0 = torch.exp(1j * 2 * np.pi * torch.rand((B, N, M), device=device))
    H_A = torch.bmm(H_batch, A0)                # (B, K, M)
    # pseudo-inverse: pinv(H_A) -> (B, M, K)
    D0 = torch.linalg.pinv(H_A)
    if not torch.is_tensor(P_BS):
        P_vec = torch.tensor(float(P_BS), device=device).view(1).repeat(B)
    else:
        P_vec = P_BS.view(-1).to(device)
    sqrtP = torch.sqrt(P_vec).view(B, 1, 1)
    norm_AD = torch.linalg.norm(torch.bmm(A0, D0), ord='fro', dim=(1, 2), keepdim=True)
    D0 = D0 * (sqrtP / (norm_AD + eps))
    return A0, D0


def svd_initialization_torch_batch(H_batch, N, M, K, P_BS, device=device):
    """
    Batch SVD initialization.
    H_batch: (B, K, N)
    Returns A0 (B, N, M), D0 (B, M, K)
    """
    B = H_batch.shape[0]
    eps = 1e-12
    # batched SVD: U (B,K,min), S (B,min), Vh (B,min,N)
    U, S, Vh = torch.linalg.svd(H_batch, full_matrices=False)
    # Vh has shape (B, N, N?) or (B, min, N); build V_full by conj().T appropriately:
    # safest: construct V_cols by taking Vh.conj().transpose(-2,-1) and select first M cols
    V = Vh.conj().transpose(-2, -1)  # (B, N, N) or (B, N, min)
    A0 = V[:, :, :M]
    A0 = torch.exp(1j * torch.angle(A0))
    H_A = torch.bmm(H_batch, A0)   # (B, K, M)
    D0 = torch.linalg.pinv(H_A)    # (B, M, K)
    if not torch.is_tensor(P_BS):
        P_vec = torch.tensor(float(P_BS), device=device).view(1).repeat(B)
    else:
        P_vec = P_BS.view(-1).to(device)
    sqrtP = torch.sqrt(P_vec).view(B, 1, 1)
    norm_AD = torch.linalg.norm(torch.bmm(A0, D0), ord='fro', dim=(1, 2), keepdim=True)
    D0 = D0 * (sqrtP / (norm_AD + eps))
    return A0, D0


# ---- Batched PGA (single+batch compatible) ----
def run_pga_torch_batch(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, P_BS, device=None, print_every=10):
    """
    Batched PGA. H: (B,K,N) or (K,N). A0: (B,N,M) or (N,M). D0: (B,M,K) or (M,K).
    Returns: objective_history_avg (I_max,), objective_history_batch (I_max, B), R_batch_final, A_final, D_final
    """
    if device is None:
        device = H.device if torch.is_tensor(H) else torch.device('cpu')

    single = False
    if H.dim() == 2:
        single = True
        H = H.unsqueeze(0)
        A0 = A0.unsqueeze(0)
        D0 = D0.unsqueeze(0)
        if Psi.dim() == 2:
            Psi = Psi.unsqueeze(0)

    H = H.to(device)
    A = A0.clone().to(device)
    D = D0.clone().to(device)
    Psi = Psi.to(device)

    B = H.shape[0]
    N = A.shape[1]
    eps = 1e-12
    eta = 1.0 / float(N)

    # ensure P_BS as vector length B
    if not torch.is_tensor(P_BS):
        P_BS = torch.tensor(float(P_BS), dtype=torch.float32, device=device)
    P_BS = P_BS.to(device)
    if P_BS.dim() == 0:
        P_BS = P_BS.repeat(B)
    elif P_BS.shape[0] != B:
        P_BS = P_BS.view(-1).repeat(B)[:B]

    objective_history_batch = torch.zeros((I_max, B), dtype=torch.float32, device=device)

    # helper to compute per-sample Frobenius norm of complex tensors
    def batch_fro_norm(X):
        # X: (B, ..., ...) -> flatten per sample and compute norm
        Bx = X.shape[0]
        flat = X.reshape(Bx, -1)
        # torch.linalg.norm returns float tensor
        return torch.linalg.norm(flat, dim=1)

    for i in range(I_max):
        # inner analog updates
        A_hat = A.clone()
        for j in range(J):
            grad_R_A = gradient_R_A_torch(H, A_hat, D, sigma_n2)
            grad_tau_A = gradient_tau_A_torch(A_hat, D, Psi)

            # OPTIONAL: scale / clip gradients here if needed (see notes below)
            # Example: clip per-sample fro norms to max_norm
            # max_norm = 10.0
            # nR = batch_fro_norm(grad_R_A); nT = batch_fro_norm(grad_tau_A)
            # scale_R = torch.minimum(torch.ones_like(nR), max_norm / (nR + 1e-12))
            # scale_T = torch.minimum(torch.ones_like(nT), max_norm / (nT + 1e-12))
            # grad_R_A = grad_R_A * scale_R.view(B, 1, 1)
            # grad_tau_A = grad_tau_A * scale_T.view(B, 1, 1)

            grad_A = grad_R_A - omega * grad_tau_A
            A_hat = A_hat + mu * grad_A
            A_hat = torch.exp(1j * torch.angle(A_hat))   # unit-modulus projection

        A = A_hat.clone()

        # digital update
        grad_R_D = gradient_R_D_torch(H, A, D, sigma_n2)
        grad_tau_D = gradient_tau_D_torch(A, D, Psi)

        # print("grad_tau_D abs range:", grad_tau_D.abs().min().item(), grad_tau_D.abs().max().item())
        # print("grad_tau_D abs mean:", grad_tau_D.abs().mean().item())
        # print("grad_tau_D abs max:", grad_tau_D.abs().max().item())


        # 🔍 Debug print for D gradients (optional)
        if i == 0:
            print("grad_tau_D range:", 
                grad_tau_D.abs().min().item(), 
                grad_tau_D.abs().max().item())

        # OPTIONAL: rescale grad_tau_D if it's much larger than grad_R_D
        # nR_D = batch_fro_norm(grad_R_D); nT_D = batch_fro_norm(grad_tau_D)
        # scale_tau_D = (nR_D / (omega * nT_D + 1e-12)).clamp(max=1.0).view(B,1,1)
        # grad_tau_D = grad_tau_D * scale_tau_D

        grad_D = grad_R_D - omega * eta * grad_tau_D
        D = D + lambda_ * grad_D

        # compute diagnostics (print every few iterations)
        if (i % print_every) == 0:
            # norms of the most recently computed grads (batched)
            try:
                norm_R_A = batch_fro_norm(grad_R_A)
                norm_tau_A = batch_fro_norm(grad_tau_A)
                norm_R_D = batch_fro_norm(grad_R_D)
                norm_tau_D = batch_fro_norm(grad_tau_D)
                #print(f"iter {i:3d}: ||gR_A|| mean={norm_R_A.mean().item():.3e}, "
                 #     f"||gτ_A|| mean={norm_tau_A.mean().item():.3e}, "
                  #    f"||gR_D|| mean={norm_R_D.mean().item():.3e}, "
                   #   f"||gτ_D|| mean={norm_tau_D.mean().item():.3e}")
            except Exception as e:
                print("Diagnostics error:", e)

        # safe power projection per-sample
        AD = torch.bmm(A, D)  # (B,N,K)
        norm_AD = torch.linalg.norm(AD.reshape(B, -1), dim=1, keepdim=True)  # (B,1)
        sqrtP = torch.sqrt(P_BS).view(B, 1).to(device)
        # expand to (B,1,1) to multiply D of shape (B,M,K)
        D = D * (sqrtP.view(B, 1, 1) / (norm_AD.view(B, 1, 1) + eps))

        # compute objective
        R_batch = compute_rate_torch(H, A, D, sigma_n2)   # (B,)
        tau_batch = compute_tau_torch(A, D, Psi)         # (B,)
        objective_batch = (R_batch - omega * tau_batch).real
        objective_history_batch[i, :] = objective_batch.to(torch.float32)

    objective_history_batch_np = objective_history_batch.cpu().numpy()
    objective_history_avg = objective_history_batch_np.mean(axis=1)
    R_batch_final = compute_rate_torch(H, A, D, sigma_n2)

    if single:
        objective_history_batch_np = objective_history_batch_np[:, 0]
        R_batch_final = R_batch_final.squeeze(0)
        A = A.squeeze(0)
        D = D.squeeze(0)

    return objective_history_avg, objective_history_batch_np, R_batch_final, A, D


# ---- Zero forcing batch version ----
def compute_R_ZF_torch(H, sigma_n2, P_BS, device=device):
    """
    H: (B,K,N) or (K,N)
    Returns per-sample sum-rate(s).
    """
    single = False
    if H.dim() == 2:
        single = True
        H = H.unsqueeze(0)

    B, K, N = H.shape
    X_ZF = torch.bmm(H.conj().transpose(1, 2), torch.linalg.pinv(torch.bmm(H, H.conj().transpose(1, 2))))  # (B, N, K)
    # normalize
    if not torch.is_tensor(P_BS):
        P_BS = torch.tensor(float(P_BS), device=device)
    P_BS = P_BS.to(device)
    if P_BS.dim() == 0:
        P_BS = P_BS.repeat(B)
    sqrtP = torch.sqrt(P_BS).view(B, 1, 1)
    norms = torch.linalg.norm(X_ZF, ord='fro', dim=(1, 2), keepdim=True)
    X_ZF = X_ZF * (sqrtP / (norms + 1e-12))

    R_out = torch.zeros(B, device=device)
    for b in range(B):
        Hb = H[b]                # (K,N)
        Xb = X_ZF[b]             # (N,K)
        rb = 0.0
        for k in range(K):
            h_k = Hb[k:k+1, :]                          # (1,N)
            signal = torch.abs(h_k @ Xb[:, k:k+1])**2
            interference = torch.sum(torch.abs(h_k @ Xb)**2) - signal
            SINR = signal / (interference + sigma_n2)
            rb = rb + torch.log2(1 + SINR)
        R_out[b] = rb.real

    if single:
        return R_out.squeeze(0)
    return R_out
