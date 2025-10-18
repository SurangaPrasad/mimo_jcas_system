import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to use CuPy for GPU acceleration
try:
    import cupy as cp
    xp = cp
    linalg = cp.linalg
    print("Using GPU with CuPy")
except ImportError:
    cp = None
    xp = np
    from scipy import linalg
    print("Using CPU with NumPy")

# Set random seed for reproducibility
xp.random.seed(42)

class JCASHybridBeamforming:
    """
    Joint Communications and Sensing Hybrid Beamforming
    Based on Algorithm 1 from the paper
    """
    def __init__(self, N, M, K, P_targets=3, omega=0.3):
        """
        N: Number of antennas
        M: Number of RF chains
        K: Number of users
        P_targets: Number of sensing targets
        omega: Weight for sensing vs communications tradeoff
        """
        self.N = N
        self.M = M
        self.K = K
        self.P = P_targets
        self.omega = omega
        
    def generate_channel(self, L=20, sigma_n=1.0):
        """
        Generate mmWave channel using geometric model (eq. 16 from Sohrabi paper)
        L: Number of paths
        """
        H = xp.zeros((self.K, self.N), dtype=complex)
        
        for k in range(self.K):
            h_k = xp.zeros(self.N, dtype=complex)
            for l in range(L):
                alpha = xp.sqrt(0.5) * (xp.random.randn() + 1j * xp.random.randn())
                phi = xp.random.uniform(0, 2 * xp.pi)
                a_t = self.steering_vector(phi, self.N)
                h_k += alpha * a_t
            H[k, :] = h_k * xp.sqrt(self.N / L)
        
        self.H = H
        self.sigma_n = sigma_n
        return H
    
    def steering_vector(self, theta, N):
        """
        Generate steering vector for uniform linear array
        theta: angle in radians
        N: number of antennas
        """
        n = xp.arange(N)
        return xp.exp(1j * xp.pi * n * xp.sin(theta)) / xp.sqrt(N)
    
    def compute_benchmark_covariance(self, target_angles, delta_theta=5):
        """
        Compute benchmark covariance matrix Psi (eq. 3-4)
        target_angles: list of desired sensing angles in degrees
        delta_theta: half of mainlobe beamwidth in degrees
        """
        T = 181  # Fine angular grid from -90 to 90 degrees
        theta_grid = xp.linspace(-90, 90, T)
        
        # Define desired beampattern
        B_d = xp.zeros(T)
        for target_deg in target_angles:
            mask = xp.abs(theta_grid - target_deg) <= delta_theta
            B_d[mask] = 1.0
        
        # Convert to radians
        theta_rad = xp.deg2rad(theta_grid)
        
        # Build matrix for optimization
        A_matrix = xp.zeros((T, self.N), dtype=complex)
        for t, theta in enumerate(theta_rad):
            a_vec = self.steering_vector(theta, self.N)
            A_matrix[t, :] = a_vec.conj()
        
        # Solve for Psi (simplified version)
        # Psi = argmin sum |alpha*B_d - a^H Psi a|^2
        # Use least squares approximation
        Psi = xp.zeros((self.N, self.N), dtype=complex)
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    Psi[i, j] = self.P_BS / self.N
                else:
                    # Simplified: use desired beampattern to construct Psi
                    sum_val = 0
                    for t, theta in enumerate(theta_rad):
                        a_vec = self.steering_vector(theta, self.N)
                        sum_val += B_d[t] * a_vec[i].conj() * a_vec[j]
                    Psi[i, j] = sum_val / T
        
        # Ensure Hermitian and PSD
        Psi = (Psi + Psi.conj().T) / 2
        self.Psi = Psi
        return Psi
    
    def initialize_precoders(self, target_angles_deg):
        """
        Initialize A and D according to equation (17)
        """
        # Build matrix G = [h1, ..., hK, a(theta_d,1), ..., a(theta_d,M-K)]
        G = []
        for k in range(self.K):
            G.append(self.H[k, :])
        
        # Add steering vectors for desired sensing angles
        n_sensing = min(self.M - self.K, len(target_angles_deg))
        for p in range(n_sensing):
            theta_rad = xp.deg2rad(target_angles_deg[p])
            a_vec = self.steering_vector(theta_rad, self.N)
            G.append(a_vec)
        
        G = xp.array(G).T  # Shape: (N, M)
        
        # Initialize A(0): phases from G
        A = xp.exp(-1j * xp.angle(G[:, :self.M]))
        
        # Initialize D(0): pseudo-inverse approach
        X_ZF = linalg.pinv(self.H)  # Shape: (N, K)
        D = linalg.pinv(A) @ X_ZF
        
        # Normalize D to satisfy power constraint
        norm_factor = xp.sqrt(self.P_BS / xp.linalg.norm(A @ D, 'fro')**2)
        D = D * norm_factor
        
        return A, D
    
    def compute_sum_rate(self, A, D):
        """
        Compute communications sum rate R (equation 2)
        """
        R = 0
        for k in range(self.K):
            h_k = self.H[k, :]
            d_k = D[:, k]
            
            # Signal power
            signal = xp.abs(h_k @ A @ d_k)**2
            
            # Interference power
            interference = 0
            for kp in range(self.K):
                if kp != k:
                    d_kp = D[:, kp]
                    interference += xp.abs(h_k @ A @ d_kp)**2
            
            # SINR
            sinr = signal / (interference + self.sigma_n**2)
            R += xp.log2(1 + sinr)
        
        return R
    
    def compute_beampattern_error(self, A, D):
        """
        Compute sensing beampattern error tau (equation 3)
        """
        ADD_H = A @ D @ D.conj().T @ A.conj().T
        tau = xp.linalg.norm(ADD_H - self.Psi, 'fro')**2
        return tau
    
    def compute_gradient_A_R(self, A, D):
        """
        Compute gradient of R w.r.t. A (equation 10)
        """
        xi = 1.0 / xp.log(2)
        grad = xp.zeros_like(A, dtype=complex)
        
        V = D @ D.conj().T
        
        for k in range(self.K):
            H_k_tilde = xp.outer(self.H[k, :].conj(), self.H[k, :])
            
            # First term
            H_k_AV = H_k_tilde @ A @ V
            denom1 = xp.trace(A @ V @ A.conj().T @ H_k_tilde) + self.sigma_n**2
            term1 = xi * H_k_AV / denom1
            
            # Second term (interference)
            D_bar_k = D.copy()
            D_bar_k[:, k] = 0
            V_bar_k = D_bar_k @ D_bar_k.conj().T
            H_k_AV_bar = H_k_tilde @ A @ V_bar_k
            denom2 = xp.trace(A @ V_bar_k @ A.conj().T @ H_k_tilde) + self.sigma_n**2
            term2 = xi * H_k_AV_bar / denom2
            
            grad += term1 - term2
        
        return grad
    
    def compute_gradient_D_R(self, A, D):
        """
        Compute gradient of R w.r.t. D (equation 11)
        """
        xi = 1.0 / xp.log(2)
        grad = xp.zeros_like(D, dtype=complex)
        
        for k in range(self.K):
            H_k_bar = A.conj().T @ xp.outer(self.H[k, :].conj(), self.H[k, :]) @ A
            
            # First term
            denom1 = xp.trace(D @ D.conj().T @ H_k_bar) + self.sigma_n**2
            term1 = xi * H_k_bar @ D / denom1
            
            # Second term
            D_bar_k = D.copy()
            D_bar_k[:, k] = 0
            denom2 = xp.trace(D_bar_k @ D_bar_k.conj().T @ H_k_bar) + self.sigma_n**2
            term2 = xi * H_k_bar @ D_bar_k / denom2
            
            grad += term1 - term2
        
        return grad
    
    def compute_gradient_A_tau(self, A, D):
        """
        Compute gradient of tau w.r.t. A (equation 12)
        """
        ADD_H = A @ D @ D.conj().T @ A.conj().T
        grad = 2 * (ADD_H - self.Psi) @ A @ D @ D.conj().T
        return grad
    
    def compute_gradient_D_tau(self, A, D):
        """
        Compute gradient of tau w.r.t. D (equation 13)
        """
        ADD_H = A @ D @ D.conj().T @ A.conj().T
        grad = 2 * A.conj().T @ (ADD_H - self.Psi) @ A @ D
        return grad
    
    def project_A(self, A):
        """
        Project A onto constant modulus constraint (equation 7)
        """
        return xp.exp(1j * xp.angle(A))
    
    def project_D(self, A, D):
        """
        Normalize D to satisfy power constraint (equation 9)
        """
        norm_factor = xp.sqrt(self.P_BS / xp.linalg.norm(A @ D, 'fro')**2)
        return D * norm_factor
    
    def run_algorithm(self, I, J, mu_init=0.01, lambda_init=0.01, 
                     target_angles_deg=[-60, 0, 60], P_BS=1.0,
                     learned_step_sizes=None):
        """
        Run Algorithm 1
        I: Number of outer iterations
        J: Number of inner iterations for A
        mu_init, lambda_init: Initial step sizes
        learned_step_sizes: If provided, use these instead of fixed steps
        """
        self.P_BS = P_BS
        
        # Generate channel and benchmark covariance
        self.generate_channel()
        self.compute_benchmark_covariance(target_angles_deg)
        
        # Initialize precoders (Step 1)
        A, D = self.initialize_precoders(target_angles_deg)
        
        # Store objective values
        objectives = []
        sum_rates = []
        beam_errors = []
        
        eta = 1.0 / self.N  # Balancing weight for gradient of D
        
        # Main loop (Steps 2-11)
        for i in range(I):
            # Inner loop for updating A (Steps 3-8)
            A_hat = A.copy()
            
            for j in range(J):
                # Use learned or fixed step size
                if learned_step_sizes is not None:
                    mu = learned_step_sizes['mu'][i, j]
                else:
                    mu = mu_init
                
                # Compute gradients (Step 5)
                grad_A_R = self.compute_gradient_A_R(A_hat, D)
                grad_A_tau = self.compute_gradient_A_tau(A_hat, D)
                
                # Update A_hat (Step 6, equation 14b)
                A_hat = A_hat + mu * (grad_A_R - self.omega * grad_A_tau)
            
            # Set A and project (Steps 8)
            A = self.project_A(A_hat)
            
            # Update D (Steps 9-10)
            if learned_step_sizes is not None:
                lambda_val = learned_step_sizes['lambda'][i]
            else:
                lambda_val = lambda_init
            
            grad_D_R = self.compute_gradient_D_R(A, D)
            grad_D_tau = self.compute_gradient_D_tau(A, D)
            
            # Update D (equation 15)
            D = D + lambda_val * (grad_D_R - self.omega * eta * grad_D_tau)
            
            # Project D (equation 9)
            D = self.project_D(A, D)
            
            # Compute objective
            R = self.compute_sum_rate(A, D)
            tau = self.compute_beampattern_error(A, D)
            obj = R - self.omega * tau
            
            objectives.append(obj)
            sum_rates.append(R)
            beam_errors.append(tau)
        
        return A, D, objectives, sum_rates, beam_errors


def compare_initializations(N=64, K=4, M=4, I=120):
    """
    Recreate Figure 2: Convergence comparison with different initializations
    """
    jcas = JCASHybridBeamforming(N=N, M=M, K=K, omega=0.3)
    jcas.P_BS = 15.85  # 12 dB in linear scale (10^(12/10))
    jcas.sigma_n = 1.0
    
    target_angles = [-60, 0, 60]
    
    print("Running convergence comparisons...")
    
    # 1. Proposed initialization, J=10
    print("  Proposed init, J=10...")
    jcas.generate_channel(L=20)
    jcas.compute_benchmark_covariance(target_angles)
    A_prop, D_prop = jcas.initialize_precoders(target_angles)
    _, _, obj_prop_10, _, _ = jcas.run_algorithm(I, J=10, mu_init=0.01, 
                                                   lambda_init=0.01,
                                                   target_angles_deg=target_angles)
    
    # 2. Proposed initialization, J=20
    print("  Proposed init, J=20...")
    jcas.generate_channel(L=20)
    jcas.compute_benchmark_covariance(target_angles)
    _, _, obj_prop_20, _, _ = jcas.run_algorithm(I, J=20, mu_init=0.01,
                                                   lambda_init=0.01,
                                                   target_angles_deg=target_angles)
    
    # 3. Random initialization, J=20
    print("  Random init, J=20...")
    jcas.generate_channel(L=20)
    jcas.compute_benchmark_covariance(target_angles)
    A_rand = xp.exp(1j * xp.random.uniform(0, 2*xp.pi, (N, M)))
    D_rand = xp.random.randn(M, K) + 1j * xp.random.randn(M, K)
    D_rand = D_rand / xp.linalg.norm(D_rand, 'fro') * xp.sqrt(jcas.P_BS)
    # Run with manual initialization
    A, D = A_rand, D_rand
    obj_rand = []
    eta = 1.0 / N
    for i in range(I):
        A_hat = A.copy()
        for j in range(20):
            grad_A_R = jcas.compute_gradient_A_R(A_hat, D)
            grad_A_tau = jcas.compute_gradient_A_tau(A_hat, D)
            A_hat = A_hat + 0.01 * (grad_A_R - 0.3 * grad_A_tau)
        A = jcas.project_A(A_hat)
        grad_D_R = jcas.compute_gradient_D_R(A, D)
        grad_D_tau = jcas.compute_gradient_D_tau(A, D)
        D = D + 0.01 * (grad_D_R - 0.3 * eta * grad_D_tau)
        D = jcas.project_D(A, D)
        R = jcas.compute_sum_rate(A, D)
        tau = jcas.compute_beampattern_error(A, D)
        obj_rand.append(R - 0.3 * tau)
    
    # 4. SVD-based initialization, J=20
    print("  SVD init, J=20...")
    jcas.generate_channel(L=20)
    jcas.compute_benchmark_covariance(target_angles)
    U, S, Vh = linalg.svd(jcas.H)
    A_svd_parts = []
    for k in range(K):
        A_svd_parts.append(Vh[k, :])
    for p in range(M - K):
        theta_rad = xp.deg2rad(target_angles[p])
        A_svd_parts.append(jcas.steering_vector(theta_rad, N))
    A_svd = xp.array(A_svd_parts).T
    A_svd = xp.exp(-1j * xp.angle(A_svd))
    D_svd = linalg.pinv(A_svd) @ linalg.pinv(jcas.H)
    D_svd = D_svd / xp.linalg.norm(A_svd @ D_svd, 'fro') * xp.sqrt(jcas.P_BS)
    A, D = A_svd, D_svd
    obj_svd = []
    for i in range(I):
        A_hat = A.copy()
        for j in range(20):
            grad_A_R = jcas.compute_gradient_A_R(A_hat, D)
            grad_A_tau = jcas.compute_gradient_A_tau(A_hat, D)
            A_hat = A_hat + 0.01 * (grad_A_R - 0.3 * grad_A_tau)
        A = jcas.project_A(A_hat)
        grad_D_R = jcas.compute_gradient_D_R(A, D)
        grad_D_tau = jcas.compute_gradient_D_tau(A, D)
        D = D + 0.01 * (grad_D_R - 0.3 * eta * grad_D_tau)
        D = jcas.project_D(A, D)
        R = jcas.compute_sum_rate(A, D)
        tau = jcas.compute_beampattern_error(A, D)
        obj_svd.append(R - 0.3 * tau)
    
    # Plot Figure 2
    plt.figure(figsize=(10, 6))
    plt.plot(range(I), obj_prop_10, 'b-', linewidth=2, label='PGA, J=10, Proposed init.')
    plt.plot(range(I), obj_prop_20, 'r-', linewidth=2, label='PGA, J=20, Proposed init.')
    plt.plot(range(I), obj_rand, 'g--', linewidth=2, label='PGA, J=20, Random init.')
    plt.plot(range(I), obj_svd, 'm-.', linewidth=2, label='PGA, J=20, SVD-based init.')
    plt.xlabel('Number of iterations/layers I', fontsize=12)
    plt.ylabel('Objective value R - ωτ', fontsize=12)
    plt.title('Convergence Comparison (Fig. 2)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figure2_convergence.png', dpi=300, bbox_inches='tight')
    print("Figure 2 saved as 'figure2_convergence.png'")
    plt.show()


def performance_vs_snr(N=64, K=4, M=4, I=120):
    """
    Recreate Figure 3: Performance vs SNR
    """
    SNR_dB = xp.arange(0, 14, 2)
    target_angles = [-60, 0, 60]
    n_channels = 10  # Average over multiple channel realizations
    
    # Storage for results
    results = {
        'J1': {'R': [], 'MSE': []},
        'J10': {'R': [], 'MSE': []},
        'J20': {'R': [], 'MSE': []},
        'ZF': {'R': [], 'MSE': []}
    }
    
    print("Running SNR sweep...")
    
    for snr_db in tqdm(SNR_dB):
        P_BS = xp.power(10, snr_db / 10)  # Convert dB to linear scale (sigma_n=1)
        
        R_j1, R_j10, R_j20, R_zf = [], [], [], []
        MSE_j1, MSE_j10, MSE_j20, MSE_zf = [], [], [], []
        
        for ch in range(n_channels):
            jcas = JCASHybridBeamforming(N=N, M=M, K=K, omega=0.3)
            jcas.P_BS = P_BS
            jcas.sigma_n = 1.0
            jcas.generate_channel(L=20)
            jcas.compute_benchmark_covariance(target_angles)
            
            # J=1
            _, _, _, rates_j1, errors_j1 = jcas.run_algorithm(
                I, J=1, target_angles_deg=target_angles, P_BS=P_BS)
            R_j1.append(rates_j1[-1])
            MSE_j1.append(errors_j1[-1])
            
            # J=10
            jcas.generate_channel(L=20)
            _, _, _, rates_j10, errors_j10 = jcas.run_algorithm(
                I, J=10, target_angles_deg=target_angles, P_BS=P_BS)
            R_j10.append(rates_j10[-1])
            MSE_j10.append(errors_j10[-1])
            
            # J=20
            jcas.generate_channel(L=20)
            _, _, _, rates_j20, errors_j20 = jcas.run_algorithm(
                I, J=20, target_angles_deg=target_angles, P_BS=P_BS)
            R_j20.append(rates_j20[-1])
            MSE_j20.append(errors_j20[-1])
            
            # ZF (communications only, fully digital)
            A_zf = xp.eye(N, M)
            D_zf = linalg.pinv(A_zf) @ linalg.pinv(jcas.H)
            D_zf = D_zf / xp.linalg.norm(A_zf @ D_zf, 'fro') * xp.sqrt(P_BS)
            R_zf_val = jcas.compute_sum_rate(A_zf, D_zf)
            tau_zf = jcas.compute_beampattern_error(A_zf, D_zf)
            R_zf.append(R_zf_val)
            MSE_zf.append(tau_zf)
        
        results['J1']['R'].append(xp.mean(R_j1))
        results['J1']['MSE'].append(10 * xp.log10(xp.mean(MSE_j1)))
        results['J10']['R'].append(xp.mean(R_j10))
        results['J10']['MSE'].append(10 * xp.log10(xp.mean(MSE_j10)))
        results['J20']['R'].append(xp.mean(R_j20))
        results['J20']['MSE'].append(10 * xp.log10(xp.mean(MSE_j20)))
        results['ZF']['R'].append(xp.mean(R_zf))
        results['ZF']['MSE'].append(10 * xp.log10(xp.mean(MSE_zf)))
    
    # Plot Figure 3a: Sum Rate
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB, results['J1']['R'], 'bs-', linewidth=2, 
             markersize=8, label='UPGANet (J=1)')
    plt.plot(SNR_dB, results['J10']['R'], 'ro-', linewidth=2, 
             markersize=8, label='UPGANet (J=10)')
    plt.plot(SNR_dB, results['J20']['R'], 'gd-', linewidth=2, 
             markersize=8, label='UPGANet (J=20)')
    plt.plot(SNR_dB, results['ZF']['R'], 'k^--', linewidth=2, 
             markersize=8, label='ZF (digital, comm. only)')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('R [bits/s/Hz]', fontsize=12)
    plt.title('Communications Sum Rate (Fig. 3a)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figure3a_sumrate.png', dpi=300, bbox_inches='tight')
    print("Figure 3a saved as 'figure3a_sumrate.png'")
    plt.show()
    
    # Plot Figure 3b: Beampattern MSE
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_dB, results['J1']['MSE'], 'bs-', linewidth=2, 
             markersize=8, label='UPGANet (J=1)')
    plt.plot(SNR_dB, results['J10']['MSE'], 'ro-', linewidth=2, 
             markersize=8, label='UPGANet (J=10)')
    plt.plot(SNR_dB, results['J20']['MSE'], 'gd-', linewidth=2, 
             markersize=8, label='UPGANet (J=20)')
    plt.plot(SNR_dB, results['ZF']['MSE'], 'k^--', linewidth=2, 
             markersize=8, label='ZF (digital, comm. only)')
    plt.xlabel('SNR [dB]', fontsize=12)
    plt.ylabel('Average beampattern MSE [dB]', fontsize=12)
    plt.title('Sensing Beampattern Error (Fig. 3b)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figure3b_mse.png', dpi=300, bbox_inches='tight')
    print("Figure 3b saved as 'figure3b_mse.png'")
    plt.show()


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("JCAS Hybrid Beamforming - Algorithm 1 Implementation")
    print("="*60)
    
    # Recreate Figure 2
    print("\n--- Recreating Figure 2: Convergence ---")
    compare_initializations(N=64, K=4, M=4, I=120)
    
    # Recreate Figure 3
    print("\n--- Recreating Figure 3: Performance vs SNR ---")
    performance_vs_snr(N=64, K=4, M=4, I=120)
    
    print("\n" + "="*60)
    print("All simulations completed!")
    print("="*60)