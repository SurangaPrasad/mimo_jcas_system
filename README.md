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
d = lambda_wave / 2  


% Step 3: Generate the Channel

function H = generate_channel(N, M, L)
    % Constants (assuming half-wavelength antenna spacing)
    lambda = 1;           % Normalized wavelength
    d = lambda / 2;       % Antenna spacing
    k = 2 * pi / lambda;  % Wavenumber

    % Initialize channel
    H = complex(zeros(M, N));
    
    for l = 1:L
        % Complex Gaussian gain (Rayleigh fading)
        alpha_var = (randn + 1j * randn) / sqrt(2);
        
        % Random AoA and AoD
        phi_r = rand * 2 * pi;  % Angle of arrival
        phi_t = rand * 2 * pi;  % Angle of departure
        
        % Array response (steering) vectors
        a_r = exp(1j * k * d * (0:M-1).' * sin(phi_r)) / sqrt(M);
        a_t = exp(1j * k * d * (0:N-1).' * sin(phi_t)) / sqrt(N);
        
        % Add path contribution
        H = H + sqrt(N * M / L) * alpha_var * (a_r * a_t');
    end
end

% Step 4: Steering vector and the PSI

% Computes the steering vector for a given angle and number of antennas
function a = steering_vector(theta, N)
    lambda = 1;
    d = lambda / 2;
    k = 2 * pi / lambda;
    a = exp(1j * k * d * (0:N-1).' * sin(theta)) / sqrt(N);
end


function [Psi, alpha_var] = compute_psi(N, theta_grid, B_d, P_BS, f)
% COMPUTE_PSI_CVX  Estimate benchmark covariance matrix Psi using CVX optimization
%
% Inputs:
%   N           - Number of antennas (Nr)
%   theta_grid  - Vector of spatial grid angles [rad]
%   B_d         - Desired beampattern samples, length T
%   P_BS        - Total transmit power
%   f           - Optional constraint vectors (N x Nf), e.g., null steering directions
%
% Outputs:
%   Psi         - Optimal benchmark covariance matrix (N x N)
%   alpha       - Optimal scaling factor (scalar)
%
% This function solves:
%
%   minimize_{α, Ψ ≽ 0}  ∑_t | α·B_d(θ_t) − ā^H(θ_t)·Ψ·ā(θ_t) |²
%
%   subject to:
%       diag(Ψ) = P_BS / N
%       α ≥ 0
%       Ψ ≽ 0
%       f(:,n)' * Ψ * f(:,n) = 0 (optional null constraints)

    if nargin < 5
        f = [];  % No null constraints by default
    end

    T = length(theta_grid);
    Nr = N;

    % --- Construct steering matrix A (N x T)
    A = zeros(Nr, T);
    for t = 1:T
        A(:, t) = steering_vector(theta_grid(t), Nr);
    end

    % --- CVX optimization ---
    cvx_begin quiet
        variable R1(Nr, Nr) hermitian
        variable alpha_var nonnegative

        % Objective: minimize sum_t | α·B_d(t) − a_tᴴ·R1·a_t |²
        minimize( sum_square_abs(alpha_var .* B_d(:) - diag(A' * R1 * A)) );

        subject to
            % Equal power per antenna
            diag(R1) == (P_BS / Nr) * ones(Nr, 1);

            % Positive semi-definiteness
            R1 == semidefinite(Nr);

            % Optional null constraints
            if ~isempty(f)
                Nf = size(f, 2);
                for n = 1:Nf
                    trace(conj(f(:, n)) * transpose(f(:, n)) * R1) == 0;
                end
            end
    cvx_end

    Psi = R1;

end

Psi = compute_psi(N, theta_grid, B_d, P_BS);


function tau = compute_tau(A, D, Psi, theta_grid)

    N = size(A, 1);
    V = A * D;
    T = length(theta_grid);

    tau = 0;
    for t = 1:T
        a_theta = steering_vector(theta_grid(t), N);
        term = a_theta' * (V * V' - Psi) * a_theta;
        tau = tau + abs(term)^2;
    end

    tau = tau / T;
end

function R = compute_rate(H, A, D, sigma_n2)
    H_A = H * A;               % Effective channel
    [~, K] = size(D);          % Number of users (columns of D)
    R = 0;

    for k = 1:K
        h_k = H_A(:, k);       % Channel to user k
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

function grad_A = gradient_R_A(H, A, D, sigma_n2)

    xi = 1 / log(2);                % Conversion factor from ln() to log2()
    grad_A = complex(zeros(size(A)));  % Initialize gradient
    V = D * D';                     % Effective covariance of digital precoder
    [K, ~] = size(H);               % Number of users

    for k = 1:K
        h_k = H(k, :).';            % Column vector (Nt x 1)
        H_tilde_k = h_k * h_k';     % Outer product (Nt x Nt)

        % D_bar_k = D with column k set to zero
        D_bar_k = D;
        D_bar_k(:, k) = 0;

        V_bar_k = D_bar_k * D_bar_k';

        % Denominators
        denom1 = trace(A * V * A' * H_tilde_k) + sigma_n2;
        denom2 = trace(A * V_bar_k * A' * H_tilde_k) + sigma_n2;

        % Gradient contribution
        term1 = H_tilde_k * A * V / denom1;
        term2 = H_tilde_k * A * V_bar_k / denom2;

        grad_A = grad_A + xi * (term1 - term2);
    end
end


function grad_D = gradient_R_D(H, A, D, sigma_n2)
    xi = 1 / log(2);
    grad_D = complex(zeros(size(D)));
    [K, ~] = size(H);

    for k = 1:K
        h_k = H(k, :).';            % (Nt x 1)
        H_tilde_k = h_k * h_k';     % (Nt x Nt)

        % Effective digital-domain channel
        H_bar_k = A' * H_tilde_k * A;   % (Ns x Ns)

        % D_bar_k = D with column k set to zero
        D_bar_k = D;
        D_bar_k(:, k) = 0;

        % Denominator terms
        denom1 = trace(D * D' * H_bar_k) + sigma_n2;
        denom2 = trace(D_bar_k * D_bar_k' * H_bar_k) + sigma_n2;

        % Gradient contributions
        term1 = (H_bar_k * D) / denom1;
        term2 = (H_bar_k * D_bar_k) / denom2;

        grad_D = grad_D + xi * (term1 - term2);
    end
end

function grad_A = gradient_tau_A(A, D, Psi)
    U = A * D * D' * A';           % A D Dᴴ Aᴴ
    grad_A = 2 * (U - Psi) * A * D * D';
end

function grad_D = gradient_tau_D(A, D, Psi)
    U = A * D * D' * A';           % A D Dᴴ Aᴴ
    grad_D = 2 * A' * (U - Psi) * A * D;
end

function [A0, D0] = proposed_initialization(H, theta_d, N, M, K, P_BS)

    G = H.';                              % (N x K)
    A0 = exp(-1j * angle(G(:, 1:M)));     % Phase-only projection

    X_ZF = pinv(H);                       % Zero-forcing baseband precoder
    D0 = pinv(A0) * X_ZF;                 % Digital precoder initialization

    % Power normalization
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end

function [A0, D0] = random_initialization(N, M, H, P_BS)

    A0 = exp(1j * 2 * pi * rand(N, M));   % Random phase analog precoder
    D0 = pinv(H * A0);                    % Pseudoinverse-based digital precoder

    % Normalize transmit power
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end

function [A0, D0] = svd_initialization(H, N, M, K, P_BS)

    [~, ~, V] = svd(H, 'econ');           % H = U*S*V', V is N x N
    A0 = V(:, 1:M);                       % Take first M right singular vectors
    A0 = exp(1j * angle(A0));             % Project to unit modulus (phase-only)

    H_A = H * A0;                         % Effective channel
    % Robust pseudoinverse (handles rank deficiency)
    if rcond(H_A) < 1e-8
        D0 = pinv(H_A + 1e-6 * eye(size(H_A)));
    else
        D0 = pinv(H_A);
    end

    % Power normalization
    D0 = sqrt(P_BS) * D0 / norm(A0 * D0, 'fro');
end


function objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS)
    [K, N] = size(H);          % H is K x N
    A = A0;
    D = D0;
    objectives = zeros(1, I_max);
    eta = 1 / N;               % Balancing term for gradient magnitudes

    for i = 1:I_max
        fprintf('\n===== Outer Iteration %d / %d =====\n', i, I_max);

        % ---- Inner Loop: Analog Precoder Update ----
        A_hat = A;
        for j = 1:J
            grad_R_A = gradient_R_A(H, A_hat, D, sigma_n2);
            grad_tau_A = gradient_tau_A(A_hat, D, Psi);

            % Gradient Ascent on A (Eq. 14b)
            grad_A = grad_R_A - omega * grad_tau_A;
            A_hat = A_hat + mu * grad_A;

            % Unit-modulus projection (Eq. 7)
            A_hat = exp(1j * angle(A_hat));
        end

        A = A_hat;  % Update analog precoder

        % ---- Digital Precoder Update ----
        grad_R_D = gradient_R_D(H, A, D, sigma_n2);
        grad_tau_D = gradient_tau_D(A, D, Psi);

        % Gradient Ascent on D (Eq. 15)
        grad_D = grad_R_D - omega * eta * grad_tau_D;
        D = D + lambda_ * grad_D;

        % Power constraint projection (Eq. 9)
        D = sqrt(P_BS) * D / norm(A * D, 'fro');

        % ---- Compute Objective ----
        R = compute_rate(H, A, D, sigma_n2);
        tau = compute_tau(A, D, Psi, theta_grid);
        objective = R - omega * tau;
        objectives(i) = objective;

        fprintf('Iteration %d: R = %.4f, τ = %.4e, Objective = %.4f\n', i, R, tau, objective);
    end
end

% ============================
% Step 6: Generate Plot (MATLAB)
% ============================

% Define containers for results
results = struct( ...
    'PGA_J10_Random', [], ...
    'PGA_J10_SVD', [], ...
    'PGA_J10_Proposed', [], ...
    'PGA_J20_Random', [], ...
    'PGA_J20_SVD', [], ...
    'PGA_J20_Proposed', [], ...
    'UPGANet_J10_Proposed', [], ...
    'UPGANet_J20_Proposed', [] );

% Loop over realizations
for r = 1:num_realizations
    fprintf('\n=== Channel Realization %d / %d ===\n', r, num_realizations);
    H = generate_channel(N, M, L);  % User-defined channel generator

    for J = J_values
        fprintf('\n--- Inner iterations: J = %d ---\n', J);

        % ---------- Random Initialization ----------
        [A0, D0] = random_initialization(N, M, H, P_BS);
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(sprintf('PGA_J%d_Random', J)){r} = objectives;
        fprintf('Completed PGA with J=%d using Random Initialization\n', J);

        % ---------- SVD Initialization ----------
        [A0, D0] = svd_initialization(H, N, M, K, P_BS);
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(sprintf('PGA_J%d_SVD', J)){r} = objectives;
        fprintf('Completed PGA with J=%d using SVD Initialization\n', J);

        % ---------- Proposed Initialization ----------
        [A0, D0] = proposed_initialization(H, theta_d, N, M, K, P_BS);
        objectives = run_pga(H, A0, D0, J, I_max, mu, lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(sprintf('PGA_J%d_Proposed', J)){r} = objectives;
        fprintf('Completed PGA with J=%d using Proposed Initialization\n', J);

        % ---------- UPGANet (Simulated) ----------
        objectives = run_pga(H, A0, D0, J, I_max, 1.5*mu, 1.5*lambda_, omega, sigma_n2, Psi, theta_grid, P_BS);
        results.(sprintf('UPGANet_J%d_Proposed', J)){r} = objectives;
        fprintf('Completed UPGANet simulation with J=%d using Proposed Initialization\n', J);
    end
end

% ---------- Compute Average Results ----------
avg_results = struct();
fields = fieldnames(results);

for i = 1:length(fields)
    key = fields{i};
    val = results.(key);
    % Convert cell array to matrix and average across realizations
    mat = cell2mat(val');
    avg_results.(key) = mean(mat, 1);
end

% ---------- Plot Results ----------
figure('Position', [200, 200, 1000, 600]);
colors = {'b', 'g', 'r', 'c', 'm', 'y', 'k', [1, 0.5, 0]}; % orange = [1, 0.5, 0]
styles = {'-', '--', '-.', ':', '-', '--', '-.', ':'};
labels = { ...
    'PGA, J=10, Random init.', ...
    'PGA, J=10, SVD init.', ...
    'PGA, J=10, Proposed init.', ...
    'PGA, J=20, Random init.', ...
    'PGA, J=20, SVD init.', ...
    'PGA, J=20, Proposed init.', ...
    'UPGANet, J=10, Proposed init.', ...
    'UPGANet, J=20, Proposed init.' };

hold on;
for idx = 1:length(fields)
    key = fields{idx};
    plot(1:I_max, real(avg_results.(key)), 'Color', colors{idx}, ...
         'LineStyle', styles{idx}, 'LineWidth', 1.5, 'DisplayName', labels{idx});
end
hold off;

xlabel('Number of Outer Iterations (I)', 'FontSize', 12);
ylabel('Objective Value (R - \omega\tau)', 'FontSize', 12);
title('Convergence of PGA and UPGANet', 'FontSize', 14);
legend('Location', 'best', 'FontSize', 10);
grid on;



