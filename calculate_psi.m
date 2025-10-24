clear; clc; close all;
rng(0);

%% Parameters
N = 64;
sigma_n2 = 1;
theta_desired = [-60 0 60];
delta_theta = 5;

PBS_from_snr = @(snr_db) sigma_n2*10.^(snr_db/10);
T = 181;
theta_grid = linspace(-90,90,T).';
theta_rad = deg2rad(theta_grid);
Abar_grid = exp(1j*pi*(0:N-1)' * sin(theta_rad.'));

% Desired beam pattern Bd
Bd = zeros(T,1);
for t = 1:T
    if any(abs(theta_grid(t)-theta_desired) <= delta_theta)
        Bd(t) = 1;
    end
end

%% Precompute Ψ for SNR from 0:0.01:12 dB
SNR_dB = 0:0.1:12;
numSNR = numel(SNR_dB);
Psi_all = zeros(N,N,numSNR);  % Preallocate
fprintf('Computing %d Psi matrices (this may take time)...\n', numSNR);

for idx = 1:numSNR
    snr_db = SNR_dB(idx);
    PBS = PBS_from_snr(snr_db);
    fprintf('SNR = %.2f dB\n', snr_db);

    Psi = calculate_Psi(N, T, Abar_grid, PBS, Bd);
    Psi_all(:,:,idx) = Psi; % Append each Psi
end

% Save all Psi matrices
save('Psi_all.mat','Psi_all','SNR_dB','-v7.3');  % Use -v7.3 for large arrays
fprintf('All Psi matrices saved to Psi_all.mat\n');


%% === Helper Function ===
function Psi = calculate_Psi(N, T, Abar_grid, PBS, Bd)
    try
        cvx_begin quiet
            variable Psi(N,N) hermitian semidefinite
            variable alpha_cvx
            expression err(T)
            for tt = 1:T
                a_t = Abar_grid(:,tt);
                err(tt) = alpha_cvx * Bd(tt) - a_t' * Psi * a_t;
            end
            minimize( sum_square_abs(err) )
            subject to
                diag(Psi) == (PBS/N) * ones(N,1);
                alpha_cvx >= 0;
        cvx_end
    catch ME
        warning('CVX unavailable — using Ψ = (PBS/N) I_N');
        disp('Actual CVX error:');
        disp(ME.message);
        Psi = (PBS/N)*eye(N);
    end
end
