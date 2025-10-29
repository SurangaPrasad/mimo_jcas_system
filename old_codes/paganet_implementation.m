% Step 1: Define System and Simulation Parameters
N = 64;              % Number of BS antennas
K = 4;               % Number of users
M = 4;               % Number of RF chains
omega = 0.3;         % Tradeoff weight

I_max = 60;          % Maximum outer iterations
J_values = [10, 20]; % Inner iteration counts
SNR_dB = 12;         % SNR in dB
sigma_n2 = 1;        % Noise variance
P_BS = sigma_n2 * 10^(SNR_dB / 10);  % Transmit power
mu = 0.01;           % Step size for analog precoder
lambda_ = 0.01;      % Step size for digital precoder
L = 20;              % Number of paths for channel
num_realizations = 2;% Number of channel realizations


% Step 2: Define Sensing Parameters
P = 3;                                   % Number of desired sensing angles
theta_d = [-60, 0, 60] * pi / 180;       % Desired angles in radians
delta_theta = 5 * pi / 180;              % Half beamwidth
theta_grid = linspace(-pi/2, pi/2, 181); % Angular grid [-90, 90] degrees
B_d = zeros(1, length(theta_grid));      % Desired beampattern

for t = 1:length(theta_grid)
    theta_t = theta_grid(t);
    for p = 1:length(theta_d)
        theta_p = theta_d(p);
        if abs(theta_t - theta_p) <= delta_theta
            B_d(t) = 1;
        end
    end
end


% Wavenumber and antenna spacing
lambda_wave = 1             % Wavelength (normalized)
k = 2 * pi / lambda_wave
d = lambda_wave / 2         % Antenna spacing


% Step 3: Channel Matrix Generation (Saleh-Valenzuela Model)

function H = generate_channel(N, M, L)
    % Constants (assuming half-wavelength antenna spacing)
    lambda = 1;           % Normalized wavelength
    d = lambda / 2;       % Antenna spacing
    k = 2 * pi / lambda;  % Wavenumber

    % Initialize channel
    H = complex(zeros(M, N));
    
    for l = 1:L
        % Complex Gaussian gain (Rayleigh fading)
        alpha = (randn + 1j * randn) / sqrt(2);
        
        % Random AoA and AoD
        phi_r = rand * 2 * pi;  % Angle of arrival
        phi_t = rand * 2 * pi;  % Angle of departure
        
        % Array response (steering) vectors
        a_r = exp(1j * k * d * (0:M-1).' * sin(phi_r)) / sqrt(M);
        a_t = exp(1j * k * d * (0:N-1).' * sin(phi_t)) / sqrt(N);
        
        % Add path contribution
        H = H + sqrt(N * M / L) * alpha * (a_r * a_t');
    end
end

% Steering Vector Function
% ------------------------
% Computes the steering vector for a given angle and number of antennas
function a = steering_vector(theta, N)
    lambda = 1;
    d = lambda / 2;
    k = 2 * pi / lambda;
    a = exp(1j * k * d * (0:N-1).' * sin(theta)) / sqrt(N);
end


function [Psi, alpha_opt] = compute_psi(N, theta_grid, B_d, P_BS)

    % ----- Build steering matrix a1 (N×T) -----
    k = 2*pi;          % assuming λ = 1
    d = 0.5;           % antenna spacing (λ/2)
    T = length(theta_grid);
    a1 = zeros(N, T);
    for t = 1:T
        theta = theta_grid(t);
        a1(:,t) = exp(1j*k*d*(0:N-1).'*sin(theta)) / sqrt(N);
    end

    % ----- Desired beampattern vector -----
    Pdesired = B_d(:);    % column vector

    % ----- CVX optimization -----
    cvx_begin quiet
        variable R1(N,N) hermitian      % Benchmark covariance Ψ
        variable alpha nonnegative      % Scaling factor α
        minimize( sum_square_abs(alpha .* Pdesired - diag(a1' * R1 * a1)) )
        subject to
            % --- Feasible space S ---
            diag(R1) == (P_BS/N) * ones(N,1);   % equal per-antenna power
            R1 == hermitian_semidefinite(N);    % Ψ ≽ 0, Hermitian PSD
    cvx_end

    % ----- Outputs -----
    Psi = R1;
    alpha_opt = alpha;

end

% Example usage:
Psi = compute_psi(N, theta_grid, B_d, P_BS);

function R = compute_rate(H, A, D, sigma_n2)
    [~, K] = size(D);
    H_A = H * A;   % Effective channel
    R = 0;

    for k = 1:K
        h_k = H_A(:, k);
        signal = abs(h_k' * D(:, k))^2;
        interference = 0;
        for j = 1:K
            if j ~= k
                interference = interference + abs(h_k' * D(:, j))^2;
            end
        end
        SINR = signal / (interference + sigma_n2);
        R = R + log2(1 + SINR);
    end
end

function tau = compute_tau(A, D, Psi, theta_grid, N)
    V = A * D;
    tau = 0;

    for i = 1:length(theta_grid)
        theta = theta_grid(i);
        a_theta = steering_vector(theta, N);
        diff = a_theta' * (V * V') * a_theta - a_theta' * Psi * a_theta;
        tau = tau + abs(diff)^2;
    end

    tau = tau / length(theta_grid);
end


function grad_A = gradient_R_A(H, A, D, sigma_n2)
    xi = 1 / log(2);                         % Conversion factor for log base 2
    grad_A = complex(zeros(size(A)));        % Initialize gradient matrix

    [~, K] = size(D);                        % Number of users
    V = D * D';                              % Effective digital precoder covariance

    for k = 1:K
        % User-k effective channel outer product
        h_k = H(k, :).';                     % (N x 1) channel vector
        H_tilde_k = h_k * h_k';              % (N x N)

        % D_bar_k: same as D but with column k set to zero
        D_bar_k = D;
        D_bar_k(:, k) = 0;

        V_bar_k = D_bar_k * D_bar_k';        % (M x M)

        % Denominator terms
        denom1 = trace(A * V * A' * H_tilde_k) + sigma_n2;
        denom2 = trace(A * V_bar_k * A' * H_tilde_k) + sigma_n2;

        % Gradient contribution for user k
        term1 = H_tilde_k * A * V / denom1;
        term2 = H_tilde_k * A * V_bar_k / denom2;

        grad_A = grad_A + xi * (term1 - term2);
    end
end

function grad_D = gradient_R_D(H, A, D, sigma_n2)
    % Get dimensions
    [K, N] = size(H); % K = number of users, N = number of antennas at BS
    [M, K_D] = size(D); % M = number of RF chains

    % Check if D dimensions are consistent with K
    if K ~= K_D
        error('Dimensions of H and D are inconsistent: size(H, 1) must equal size(D, 2).');
    end

    % Conversion factor from ln() to log2()
    xi = 1 / log(2);

    % Initialize gradient matrix (M x K) with complex zeros
    grad_D = complex(zeros(M, K));

    % --- Loop over users k ---
    for k = 1:K
        % (1) Channel vector for user k
        % MATLAB uses index 1 for the first element.
        % H(k, :) is (1 x N). Transpose to get (N x 1).
        h_k = H(k, :).';                % (N x 1)
        % Conjugate transpose of h_k is h_k' in MATLAB
        H_tilde_k = h_k * h_k';         % (N x N)

        % (2) Effective digital-domain channel including analog precoder
        % MATLAB's ' is conjugate transpose (equivalent to .conj().T in numpy)
        H_bar_k = A' * H_tilde_k * A;   % (M x M)

        % (3) D_bar_k = D with k-th column set to zero
        D_bar_k = D;
        D_bar_k(:, k) = 0.0;

        % (4) Compute denominator terms (trace parts)
        % D' is conjugate transpose
        denom1 = trace(D * D' * H_bar_k) + sigma_n2;
        denom2 = trace(D_bar_k * D_bar_k' * H_bar_k) + sigma_n2;

        % (5) Compute gradient contributions
        term1 = (H_bar_k * D) / denom1;
        term2 = (H_bar_k * D_bar_k) / denom2;

        % (6) Accumulate total gradient
        grad_D = grad_D + xi * (term1 - term2);
    end
end

function grad_A = gradient_tau_A(A, D, Psi)
    U = A * D * D' * A';

    % grad_A = 2 * (U - Psi) * A * D * D'
    grad_A = 2 * (U - Psi) * A * D * D';
end


function grad_D = gradient_tau_D(A, D, Psi)
    U = A * D * D' * A';

    % grad_D = 2 * A' * (U - Psi) * A * D
    grad_D = 2 * A' * (U - Psi) * A * D;
end


function [A0, D0] = proposed_initialization(H, theta_d, N, M, K, P_BS)
    G = H.';

    % A0 = exp(-1j * angle(G))(:, 1:M)
    % A0 is the phase of G, projected to unit modulus, taking the first M columns.
    A0 = exp(-1j * angle(G));
    A0 = A0(:, 1:M);

    % X_ZF = pinv(H)
    X_ZF = pinv(H);

    % D0 = pinv(A0) * X_ZF
    D0 = pinv(A0) * X_ZF;

    % Normalize D0: D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro')
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end

function [A0, D0] = random_initialization(N, M, H, P_BS)
    A0 = exp(1j * (rand(N, M) * 2 * pi));

    % D0 = pinv(H * A0)
    D0 = pinv(H * A0);

    % Normalize D0: D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro')
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end

function [A0, D0] = svd_initialization(H, N, M, K, P_BS)
    [~, ~, V] = svd(H, 'econ');

    % A0 = V(:, 1:M) (Take first M columns of V)
    % V is N x K, Vh.T (Python) is V (MATLAB).
    A0 = V(:, 1:M);

    % A0 = exp(1j * angle(A0)) (Project to unit modulus)
    A0 = exp(1j * angle(A0));

    % H_A = H * A0 (Shape: K x M)
    H_A = H * A0;

    % Try-catch block for regularization
    % Python's try/except block is mapped to MATLAB's try/catch
    try
        D0 = pinv(H_A); % Pseudoinverse of H * A0
    catch ME
        % Check if the error is due to singularity (LinAlgError in Python)
        % For simplicity, we just apply regularization on error, similar to the Python logic.
        warning('SVD_INITIALIZATION:PInvRegularization', ...
                'P-inverse failed, applying regularization.');
        D0 = pinv(H_A + 1e-6 * eye(M)); % Regularization for stability
    end

    % Normalize D0: D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro')
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end




function objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS)

    % Initialize precoders
    A = A0;
    D = D0;

    % Initialize objective storage
    objectives = zeros(1, I_max);
    
    % Balancing term for gradient magnitudes (based on Python code)
    eta = 1 / N;

    for i = 1:I_max
        fprintf('\n===== Outer Iteration %d/%d =====\n', i, I_max);

        % ---- Inner Loop: Analog Precoder Update (A) ----
        A_hat = A;
        for j = 1:J
            % Calculate gradients
            grad_R_A = gradient_R_A(H, A_hat, D, sigma_n2);
            grad_tau_A = gradient_tau_A(A_hat, D, Psi);

            % Eq. (14b): Gradient Ascent on A
            grad_A = grad_R_A - omega * grad_tau_A;
            A_hat = A_hat + mu * grad_A;

            % Eq. (7): Unit Modulus Projection
            % MATLAB's angle() returns the phase angle in radians
            A_hat = exp(1j * angle(A_hat));
        end

        % Set final A after J inner updates
        A = A_hat;

        % ---- Outer Loop: Digital Precoder Update (D) ----
        % Calculate gradients
        grad_R_D = gradient_R_D(H, A, D, sigma_n2);
        grad_tau_D = gradient_tau_D(A, D, Psi);

        % Eq. (15): Gradient Ascent on D
        grad_D = grad_R_D - omega * eta * grad_tau_D;
        D = D + lambda_ * grad_D;

        % Eq. (9): Power Constraint Projection
        % MATLAB's norm(..., 'fro') computes the Frobenius norm
        D = sqrt(P_BS) * D / norm(A * D, 'fro');

        % ---- Compute Objective (Eq. 5a) ----
        % Note: Assumes P_BS is passed to compute_rate if needed for normalization
        R = compute_rate(H, A, D, sigma_n2);
        tau = compute_tau(A, D, Psi, theta_grid);
        objective = R - omega * tau;
        objectives(i) = objective;

        fprintf('Iteration %d: R = %.4f, tau = %.4e, Objective = %.4f\n', i, R, tau, objective);
    end
end


% --- Step 6: Initialize Results Struct ---

% Initialize a structure where each field holds a cell array to store results
% from 'num_realizations' objective vectors (each of length I_max).
results = struct(...
    'PGA_J10_Random', cell(1, num_realizations), ...
    'PGA_J10_SVD', cell(1, num_realizations), ...
    'PGA_J10_Proposed', cell(1, num_realizations), ...
    'PGA_J20_Random', cell(1, num_realizations), ...
    'PGA_J20_SVD', cell(1, num_realizations), ...
    'PGA_J20_Proposed', cell(1, num_realizations), ...
    'UPGANet_J10_Proposed', cell(1, num_realizations), ...
    'UPGANet_J20_Proposed', cell(1, num_realizations));

% --- Run Simulation Loops ---
for r = 1:num_realizations
    % Generate channel realization
    % Assuming generate_channel is available and returns H (K x N)
    H = generate_channel(N, M, L); 
    
    fprintf('\nStarting Realization %d/%d\n', r, num_realizations);

    for j_idx = 1:length(J_values)
        J = J_values(j_idx);
        J_str = num2str(J);
        
        % 1. Random Initialization
        [A0, D0] = random_initialization(N, M, H, P_BS);
        % Note: P_BS is passed to run_pga as per the converted function signature
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(['PGA_J', J_str, '_Random']){r} = objectives;
        fprintf('Completed PGA with J=%d using Random Initialization\n', J);
        
        % 2. SVD Initialization
        [A0, D0] = svd_initialization(H, N, M, K, P_BS);
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(['PGA_J', J_str, '_SVD']){r} = objectives;
        fprintf('Completed PGA with J=%d using SVD Initialization\n', J);

        % 3. Proposed Initialization
        [A0, D0] = proposed_initialization(H, theta_d, N, M, K, P_BS);
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(['PGA_J', J_str, '_Proposed']){r} = objectives;
        fprintf('Completed PGA with J=%d using Proposed Initialization\n', J);
        
        % 4. UPGANet (simulated with fixed step sizes)
        % Using 1.5x step size to simulate optimized step size
        objectives = run_pga(H, A0, D0, J, I_max, mu * 1.5, lambda_ * 1.5, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(['UPGANet_J', J_str, '_Proposed']){r} = objectives;
        fprintf('Completed UPGANet simulation with J=%d using Proposed Initialization\n', J);
    end
end

% --- Average Results ---
field_names = fieldnames(results);
avg_results = struct();

for f = 1:length(field_names)
    key = field_names{f};
    data_cell = results.(key);
    
    % Convert cell array of row vectors (objectives) into a matrix
    % where each row is a realization (num_realizations x I_max)
    data_matrix = cell2mat(data_cell.'); 
    
    % Average across the first dimension (realizations)
    avg_results.(key) = mean(data_matrix, 1);
end

% --- Plotting ---

figure('Position', [100, 100, 800, 500]); % Set figure size
hold on; % Allow multiple plots on the same axes

% Define plot styles and labels
colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [0.93, 0.69, 0.13]}; % Last color is 'orange'
styles = {'-', '--', '-.', ':', '-', '--', '-.', ':'};
labels = {
    'PGA, J=10, Random init.',
    'PGA, J=10, SVD init.',
    'PGA, J=10, Proposed init.',
    'PGA, J=20, Random init.',
    'PGA, J=20, SVD init.',
    'PGA, J=20, Proposed init.',
    'UPGANet, J=10, Proposed init.',
    'UPGANet, J=20, Proposed init.'
};

% Ensure all results are plotted in the same order as the labels
plot_keys = {
    'PGA_J10_Random', 'PGA_J10_SVD', 'PGA_J10_Proposed', ...
    'PGA_J20_Random', 'PGA_J20_SVD', 'PGA_J20_Proposed', ...
    'UPGANet_J10_Proposed', 'UPGANet_J20_Proposed'
};

% Plot data
for idx = 1:length(plot_keys)
    key = plot_keys{idx};
    % The X-axis should be 1 to I_max
    plot(1:I_max, avg_results.(key), ...
         'Color', colors{idx}, ...
         'LineStyle', styles{idx}, ...
         'DisplayName', labels{idx}, ...
         'LineWidth', 1.5);
end

% Set labels, title, and grid
xlabel('Number of Outer Iterations (I)');
% \omega is the LaTeX command for the Greek letter omega
ylabel('Objective Value (R - \omega\tau)'); 
title('Convergence of PGA and UPGANet');
legend('Location', 'best');
grid on;
hold off;

