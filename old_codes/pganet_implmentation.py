
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import uuid

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Define System and Simulation Parameters
N = 64  # Number of BS antennas
K = 4   # Number of users
M = 4   # Number of RF chains
omega = 0.3  # Tradeoff weight
I_max = 120  # Maximum outer iterations
J_values = [10, 20]  # Inner iteration counts
SNR_dB = 12  # SNR in dB
sigma_n2 = 1  # Noise variance
P_BS = sigma_n2 * 10**(SNR_dB / 10)  # Transmit power
mu = 0.01  # Step size for analog precoder
lambda_ = 0.01  # Step size for digital precoder
L = 20  # Number of paths for channel
num_realizations = 100  # Number of channel realizations

# Step 2: Define Sensing Parameters
P = 3  # Number of desired sensing angles
theta_d = np.array([-60, 0, 60]) * np.pi / 180  # Desired angles in radians
delta_theta = 5 * np.pi / 180  # Half beamwidth
theta_grid = np.linspace(-np.pi / 2, np.pi / 2, 181)  # Angular grid [-90, 90] degrees
B_d = np.zeros(len(theta_grid))  # Desired beampattern
for t, theta_t in enumerate(theta_grid):
    for theta_p in theta_d:
        if abs(theta_t - theta_p) <= delta_theta:
            B_d[t] = 1

# Wavenumber and antenna spacing
lambda_wave = 1  # Wavelength (normalized)
k = 2 * np.pi / lambda_wave
d = lambda_wave / 2  # Antenna spacing

# Step 3: Channel Matrix Generation (Saleh-Valenzuela Model)
def generate_channel(N, M, L):
    H = np.zeros((M, N), dtype=complex)
    for _ in range(L):
        alpha = np.random.normal(0, 1, 2).view(complex)[0] / np.sqrt(2)  # Complex gain
        phi_r = np.random.uniform(0, 2 * np.pi)  # Angle of arrival
        phi_t = np.random.uniform(0, 2 * np.pi)  # Angle of departure
        a_r = np.exp(1j * k * d * np.arange(M) * np.sin(phi_r)) / np.sqrt(M)
        a_t = np.exp(1j * k * d * np.arange(N) * np.sin(phi_t)) / np.sqrt(N)
        H += np.sqrt(N * M / L) * alpha * np.outer(a_r, a_t.conj())
    return H

# Steering vector function
def steering_vector(theta, N):
    return np.exp(1j * k * d * np.arange(N) * np.sin(theta)) / np.sqrt(N)

# Compute benchmark covariance matrix Psi
def compute_psi(N, theta_grid, B_d):
    T = len(theta_grid)
    A_theta = np.array([steering_vector(theta, N) for theta in theta_grid]).T  # N x T
    Psi = np.zeros((N, N), dtype=complex)
    for t in range(T):
        a_t = A_theta[:, t]
        Psi += B_d[t] * np.outer(a_t, a_t.conj())
    Psi = Psi / np.trace(Psi) * N  # Normalize to satisfy trace constraint
    return Psi

Psi = compute_psi(N, theta_grid, B_d)

# Compute communication rate R
def compute_rate(H, A, D, sigma_n2):
    H_A = H @ A  # Effective channel
    R = 0
    for k in range(K):
        h_k = H_A[:, k]
        signal = np.abs(h_k.conj().T @ D[:, k])**2
        interference = sum(np.abs(h_k.conj().T @ D[:, j])**2 for j in range(K) if j != k)
        SINR = signal / (interference + sigma_n2)
        R += np.log2(1 + SINR)
    return R

# Compute sensing error tau
def compute_tau(A, D, Psi, theta_grid):
    V = A @ D
    tau = 0
    for theta in theta_grid:
        a_theta = steering_vector(theta, N)
        tau += np.abs(a_theta.conj().T @ V @ V.conj().T @ a_theta - a_theta.conj().T @ Psi @ a_theta)**2
    return tau / len(theta_grid)

# Gradients (simplified for demonstration)
def gradient_R_A(H, A, D, sigma_n2):
    # Placeholder: Actual gradient requires equations (10), approximated here
    H_A = H @ A
    grad = np.zeros_like(A, dtype=complex)
    for k in range(K):
        h_k = H_A[:, k]
        d_k = D[:, k]
        signal = h_k.conj().T @ d_k
        interference = sum((h_k.conj().T @ D[:, j]) * D[:, j] for j in range(K) if j != k)
        grad += np.outer(H.conj().T @ (signal * d_k - interference), d_k.conj()) / (sigma_n2 + sum(np.abs(h_k.conj().T @ D[:, j])**2 for j in range(K)))
    return grad

def gradient_R_D(H, A, D, sigma_n2):
    # Placeholder: Actual gradient requires equation (11)
    H_A = H @ A
    grad = np.zeros_like(D, dtype=complex)
    for k in range(K):
        h_k = H_A[:, k]
        signal = h_k.conj().T @ D[:, k]
        interference = sum((h_k.conj().T @ D[:, j]) * D[:, j] for j in range(K) if j != k)
        grad[:, k] = H_A.conj().T @ h_k * signal / (sigma_n2 + sum(np.abs(h_k.conj().T @ D[:, j])**2 for j in range(K)))
    return grad

def gradient_tau_A(A, D, Psi, theta_grid):
    # Placeholder: Actual gradient requires equation (12)
    V = A @ D
    grad = np.zeros_like(A, dtype=complex)
    for theta in theta_grid:
        a_theta = steering_vector(theta, N)
        error = (a_theta.conj().T @ V @ V.conj().T @ a_theta - a_theta.conj().T @ Psi @ a_theta)
        grad += error * (np.outer(a_theta, (V.conj().T @ a_theta).conj()) @ D.conj().T)
    return grad / len(theta_grid)

def gradient_tau_D(A, D, Psi, theta_grid):
    # Placeholder: Actual gradient requires equation (13)
    V = A @ D
    grad = np.zeros_like(D, dtype=complex)
    for theta in theta_grid:
        a_theta = steering_vector(theta, N)
        error = (a_theta.conj().T @ V @ V.conj().T @ a_theta - a_theta.conj().T @ Psi @ a_theta)
        grad += error * (A.conj().T @ a_theta @ (a_theta.conj().T @ V).conj())
    return grad / len(theta_grid)

# Step 4: Initialization Strategies
def proposed_initialization(H, theta_d, N, M, K, P_BS):
    G = np.array([H[k, :] for k in range(K)]).T  # N x K
    A0 = np.exp(-1j * np.angle(G))[:, :M]
    X_ZF = np.linalg.pinv(H)
    D0 = np.linalg.pinv(A0) @ X_ZF
    D0 = np.sqrt(P_BS) * D0 / np.linalg.norm(A0 @ D0, 'fro')
    return A0, D0

def random_initialization(N, M, H, P_BS):
    A0 = np.exp(1j * np.random.uniform(0, 2 * np.pi, (N, M)))
    D0 = np.linalg.pinv(H @ A0)
    D0 = np.sqrt(P_BS) * D0 / np.linalg.norm(A0 @ D0, 'fro')
    return A0, D0

def svd_initialization(H, N, M, K, P_BS):
    U, _, _ = svd(H, full_matrices=False)
    A0 = U[:, :K]
    A0 = np.exp(1j * np.angle(A0))
    D0 = np.linalg.pinv(H @ A0)
    D0 = np.sqrt(P_BS) * D0 / np.linalg.norm(A0 @ D0, 'fro')
    return A0, D0

# Step 5: PGA Algorithm
def run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid):
    A = A0.copy()
    D = D0.copy()
    objectives = []
    eta = 1 / N
    for i in range(I_max):
        # Update A
        A_hat = A.copy()
        for j in range(J):
            grad_A = gradient_R_A(H, A_hat, D, sigma_n2) - omega * gradient_tau_A(A_hat, D, Psi, theta_grid)
            A_hat = A_hat + mu * grad_A
            A_hat = np.exp(1j * np.angle(A_hat))  # Project to unit modulus
        A = A_hat
        # Update D
        grad_D = gradient_R_D(H, A, D, sigma_n2) - omega * eta * gradient_tau_D(A, D, Psi, theta_grid)
        D = D + lambda_ * grad_D
        D = np.sqrt(P_BS) * D / np.linalg.norm(A @ D, 'fro')  # Power constraint
        # Compute objective
        R = compute_rate(H, A, D, sigma_n2)
        tau = compute_tau(A, D, Psi, theta_grid)
        objectives.append(R - omega * tau)
    return objectives

# Step 6: Generate Plot
results = {
    'PGA_J10_Random': [],
    'PGA_J10_SVD': [],
    'PGA_J10_Proposed': [],
    'PGA_J20_Random': [],
    'PGA_J20_SVD': [],
    'PGA_J20_Proposed': [],
    'UPGANet_J10_Proposed': [],  # Simulated with fixed step sizes
    'UPGANet_J20_Proposed': []   # Simulated with fixed step sizes
}

for _ in range(num_realizations):
    H = generate_channel(N, M, L)
    for J in J_values:
        # Random Initialization
        A0, D0 = random_initialization(N, M, H, P_BS)
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid)
        results[f'PGA_J{J}_Random'].append(objectives)
        
        # SVD Initialization
        A0, D0 = svd_initialization(H, N, M, K, P_BS)
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid)
        results[f'PGA_J{J}_SVD'].append(objectives)
        
        # Proposed Initialization
        A0, D0 = proposed_initialization(H, theta_d, N, M, K, P_BS)
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid)
        results[f'PGA_J{J}_Proposed'].append(objectives)
        
        # UPGANet (simulated with fixed step sizes)
        objectives = run_pga(H, A0, D0, J, I_max, mu * 1.5, lambda_ * 1.5, omega, sigma_n2, Psi, theta_grid)  # Adjusted step sizes
        results[f'UPGANet_J{J}_Proposed'].append(objectives)

# Average results
avg_results = {key: np.mean(np.array(val), axis=0) for key, val in results.items()}

# Plotting
plt.figure(figsize=(10, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
labels = [
    'PGA, J=10, Random init.',
    'PGA, J=10, SVD init.',
    'PGA, J=10, Proposed init.',
    'PGA, J=20, Random init.',
    'PGA, J=20, SVD init.',
    'PGA, J=20, Proposed init.',
    'UPGANet, J=10, Proposed init.',
    'UPGANet, J=20, Proposed init.'
]
for idx, (key, data) in enumerate(avg_results.items()):
    plt.plot(range(I_max), data, color=colors[idx], linestyle=styles[idx], label=labels[idx])
plt.xlabel('Number of Outer Iterations (I)')
plt.ylabel('Objective Value (R - ωτ)')
plt.title('Convergence of PGA and UPGANet')
plt.legend()
plt.grid(True)
plt.show()