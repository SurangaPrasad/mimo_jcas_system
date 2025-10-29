clear; close all; clc; rng(0);

%% ---------------- Simulation Parameters ----------------
N = 64;  
M = 4;  
K = 4;
P = 3; 
L = 20;
theta_desired = [-60 0 60];  
delta_theta = 5;
omega = 0.3;  
sigma_n2 = 1;
Imax = 120;   
mu_const = 0.05;  
lambda_const = 0.05;
eta = 1/N;    
J_values = [1 10 20];
SNR_dB = 0:2:12;
PBS_from_snr = @(snr_db) sigma_n2*10.^(snr_db/10);
T = 181;  
theta_grid = linspace(-90,90,T).';
num_realizations = 10;   % set 100–1000 for final averaging

%% Steering vectors and desired beampattern
theta_rad = deg2rad(theta_grid);
Abar_grid = exp(1j*pi*(0:N-1)' * sin(theta_rad.'));
Bd = zeros(T,1);
for t=1:T
    if any(abs(theta_grid(t)-theta_desired)<=delta_theta)
        Bd(t)=1;
    end
end

%% Preallocate
numSNR = length(SNR_dB);
R_PGA = zeros(1,numSNR);
R_UPG1 = R_PGA; R_UPG10 = R_PGA; R_UPG20 = R_PGA; R_ZF = R_PGA;
MSE_PGA = R_PGA; MSE_UPG1 = R_PGA; MSE_UPG10 = R_PGA; MSE_UPG20 = R_PGA; MSE_ZF = R_PGA;
beamSnap = struct();

fprintf('Running %d realizations × %d SNRs\n',num_realizations,numSNR);
for sIdx = 1:numSNR
    snr_db = SNR_dB(sIdx);
    PBS = PBS_from_snr(snr_db);
    fprintf('SNR = %2d dB\n',snr_db);

    % --- Compute benchmark covariance Ψ (Eq. 4) ---
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


    % --- Averages over channel realizations ---
    rPGA=0; rU1=0; rU10=0; rU20=0; rZF=0;
    mPGA=0; mU1=0; mU10=0; mU20=0; mZF=0;

    for rr = 1:num_realizations
        % Random narrowband mmWave channel
        H = generate_channel(N, M, L);


        % Initialization (Eq. 17)
        steering_des = exp(1j*pi*(0:N-1)'.*sin(deg2rad(theta_desired(1:M-K))));
        G = [H' steering_des];
        A0 = exp(-1j*angle(G(:,1:M)));
        
        XZF = pinv(H);
        % XZF = XZF.'; 

        D0 = pinv(A0)*XZF;
        D0 = D0*sqrt(PBS)/norm(A0*D0,'fro');

        % ----- Conventional PGA -----
        [A_pga,D_pga] = run_PGA(A0,D0,H,Psi,PBS,omega,eta,mu_const,lambda_const,Imax,1,K);
        [rTmp,mTmp,bTmp] = metrics(A_pga,D_pga,H,Bd,Abar_grid,sigma_n2);
        rPGA=rPGA+rTmp; 
        mPGA=mPGA+mTmp;
        if sIdx==numSNR && rr==1, beamSnap.PGA=bTmp; end

        % ----- UPGANet J=1,10,20 (fixed steps substitute) -----
        for Jv = J_values
            [A_u,D_u] = run_UPGANet(A0,D0,H,Psi,PBS,omega,eta,mu_const,lambda_const,Imax,Jv,K);
            [rTmp,mTmp,bTmp] = metrics(A_u,D_u,H,Bd,Abar_grid,sigma_n2);
            switch Jv
                case 1,  rU1=rU1+rTmp;  
                         mU1=mU1+mTmp;
                         if sIdx==numSNR && rr==1, beamSnap.UPG1=bTmp; end
                case 10, rU10=rU10+rTmp; 
                         mU10=mU10+mTmp;
                         if sIdx==numSNR && rr==1, beamSnap.UPG10=bTmp; end
                case 20, rU20=rU20+rTmp; 
                         mU20=mU20+mTmp;
                         if sIdx==numSNR && rr==1, beamSnap.UPG20=bTmp; end
            end
        end

        % ----- ZF (digital, communications only) -----
        Xfd = H'*pinv(H*H');  % Zero-forcing precoder
        Xfd = Xfd*sqrt(PBS)/norm(Xfd,'fro');
        % Compute sum rate for ZF
        rZF_tmp = 0;
        for k=1:K
            hk = H(k,:)';
            num = abs(hk'*Xfd(:,k))^2;
            den = sigma_n2 + sum(abs(hk'*Xfd).^2) - num;
            rZF_tmp = rZF_tmp + log2(1 + num/den);
        end
        rZF = rZF + rZF_tmp;
        
        % Beampattern for ZF
        Psi_zf = Xfd*Xfd';
        BPz = abs(diag(Abar_grid'*Psi_zf*Abar_grid));
        mZF = mZF + 10*log10(mean(abs(Bd-BPz).^2)+1e-12);
        if sIdx==numSNR && rr==1, beamSnap.ZF=BPz/max(BPz); end
    end

    % averages
    R_PGA(sIdx)=rPGA/num_realizations;  MSE_PGA(sIdx)=mPGA/num_realizations;
    R_UPG1(sIdx)=rU1/num_realizations;  MSE_UPG1(sIdx)=mU1/num_realizations;
    R_UPG10(sIdx)=rU10/num_realizations;MSE_UPG10(sIdx)=mU10/num_realizations;
    R_UPG20(sIdx)=rU20/num_realizations;MSE_UPG20(sIdx)=mU20/num_realizations;
    R_ZF(sIdx)=rZF/num_realizations;    MSE_ZF(sIdx)=mZF/num_realizations;
end

%% ---------------- Plot Figure 3 ----------------
figure('Color','w','Position',[100 100 1000 600]);

subplot(2,2,1); hold on; grid on;
plot(SNR_dB,R_UPG1,'-o'); 
plot(SNR_dB,R_UPG10,'-s'); 
plot(SNR_dB,R_UPG20,'-^');
plot(SNR_dB,R_PGA,'--');  
plot(SNR_dB,R_ZF,':');
xlabel('SNR [dB]'); 
ylabel('R [bits/s/Hz]');
legend('UPGANet J=1','UPGANet J=10','UPGANet J=20','PGA','ZF (digital)');
% legend('UPGANet J=1','UPGANet J=10','UPGANet J=20','PGA');
% legend('PGA')
title('(a) Communications sum rate');

subplot(2,2,2); hold on; grid on;
plot(SNR_dB,MSE_UPG1,'-o'); plot(SNR_dB,MSE_UPG10,'-s'); plot(SNR_dB,MSE_UPG20,'-^');
plot(SNR_dB,MSE_PGA,'--'); plot(SNR_dB,MSE_ZF,':');
xlabel('SNR [dB]'); ylabel('Avg beampattern MSE [dB]');
legend('UPGANet J=1','UPGANet J=10','UPGANet J=20','PGA','ZF (digital)');
title('(b) Average beampattern MSE');

subplot(2,2,[3 4]); hold on; grid on;
plot(theta_grid,Bd,'k','LineWidth',1.5);
fields={'UPG20','UPG10','UPG1','PGA','ZF'};
% fields={'UPG20','UPG10','UPG1','PGA'};
for f=fields
    if isfield(beamSnap,f{1})
        plot(theta_grid,10*log10(beamSnap.(f{1})+1e-12),'LineWidth',1.3);
    end
end
xlabel('Angle (°)'); ylabel('Normalized sensing beampattern [dB]');
legend(['Benchmark Bd',fields],'Location','Best');
title('(c) Sensing beampattern @ highest SNR');
sgtitle('Figure 3 — UPGANet vs PGA vs ZF');

disp('Figure 3 complete.');

%% ============================================================
%                Local helper functions
% ============================================================

function [A,D] = run_PGA(A0,D0,H,Psi,PBS,omega,eta,mu,lambda,Imax,J,K)
    A=A0; D=D0;
    for i=1:Imax
        Ahat=A;
        for j=1:J
            Ahat = Ahat + mu*(gradA_R(Ahat,D,H,K)-omega*gradA_tau(Ahat,D,Psi));
            Ahat = exp(1j*angle(Ahat)); % unit-modulus projection
        end
        A=Ahat;
        D = D + lambda*(gradD_R(A,D,H,K)-omega*eta*gradD_tau(A,D,Psi));
        D = D*sqrt(PBS)/norm(A*D,'fro');
    end
end

function [A,D] = run_UPGANet(A0,D0,H,Psi,PBS,omega,eta,mu_train,lambda_train,Imax,J,K)
    % Initialization
    A = A0; 
    D = D0;
    objHistory = zeros(Imax,1);

    % Handle cases where learned step sizes are scalars or vectors
    isLearned = (numel(mu_train) > 1) || (numel(lambda_train) > 1);

    % ===== Outer Iterations =====
    for i = 1:Imax
        % ---- Inner Loop: Analog Precoder Update ----
        Ahat = A;
        for j = 1:J
            % Select layer-specific step size μ(i,j)
            if isLearned
                idx = (i-1)*J + j;
                if idx <= numel(mu_train)
                    mu_ij = mu_train(idx);
                else
                    mu_ij = mu_train(end);
                end
            else
                mu_ij = mu_train;
            end

            % Compute gradients (Equations 10 & 12)
            gradR_A = gradA_R(Ahat,D,H,K);
            gradTau_A = gradA_tau(Ahat,D,Psi);

            % Gradient ascent update (Eq. 14b)
            Ahat = Ahat + mu_ij * (gradR_A - omega * gradTau_A);

            % Unit-modulus projection (Eq. 7)
            Ahat = exp(1j * angle(Ahat));
        end
        A = Ahat;  % finalize analog precoder

        % ---- Digital Precoder Update ----
        % Select layer-specific step size λ(i)
        if isLearned
            if i <= numel(lambda_train)
                lambda_i = lambda_train(i);
            else
                lambda_i = lambda_train(end);
            end
        else
            lambda_i = lambda_train;
        end

        % Compute gradients (Equations 11 & 13)
        gradR_D = gradD_R(A,D,H,K);
        gradTau_D = gradD_tau(A,D,Psi);

        % Gradient ascent update (Eq. 15)
        D = D + lambda_i * (gradR_D - omega * eta * gradTau_D);

        % Power constraint projection (Eq. 9)
        D = D * sqrt(PBS) / norm(A*D,'fro');

        % % ---- Track Objective (Eq. 5a) ----
        % Rval = compute_rate(H,A,D);
        % tauval = compute_tau(A,D,Psi);
        % objHistory(i) = real(Rval - omega * tauval);
    end
end


function gA = gradA_R(A,D,H,K)

    sigma_n2=1;
    xi=1/log(2); 
    V=D*D'; 
    gA=zeros(size(A));
    for k=1:K
        hk=H(k,:)'; 
        Hk=hk*hk';
        Dkbar=D; Dkbar(:,k)=0; Vkbar=Dkbar*Dkbar';
        gA = gA + xi*Hk*A*V/(trace(A*V*A'*Hk)+sigma_n2) ...
                  - xi*Hk*A*Vkbar/(trace(A*Vkbar*A'*Hk)+sigma_n2);
    end
end

function gD = gradD_R(A,D,H,K)
    sigma_n2=1;
    xi=1/log(2); 
    gD=zeros(size(D));
    for k=1:K
        hk=H(k,:)'; 
        Hbar=A'*hk*hk'*A;
        Dkbar=D; Dkbar(:,k)=0;
        gD = gD + xi*Hbar*D/(trace(D*D'*Hbar)+sigma_n2) ...
                  - xi*Hbar*Dkbar/(trace(Dkbar*Dkbar'*Hbar)+sigma_n2);
    end
end

function gA = gradA_tau(A,D,Psi)
    U=A*D*D'*A'; 
    gA=2*(U-Psi)*A*(D*D');
end

function gD = gradD_tau(A,D,Psi)
    U=A*D*D'*A'; 
    gD=2*(A'*(U-Psi)*A)*D;
end

function [Rval,MSEdB,beamp] = metrics(A,D,H,Bd,Abar,sigma2)
    PsiAD=A*(D*D')*A'; 
    Rval=0; K=size(H,1);
    for k=1:K
        hk=H(k,:)';
        num=abs(hk'*A*D(:,k))^2;
        den=sigma2+sum(abs(hk'*A*D).^2)-num;
        Rval=Rval+log2(1+num/den);
    end
    beamp=abs(diag(Abar'*PsiAD*Abar));
    MSEdB=10*log10(mean(abs(Bd-beamp).^2)+1e-12);
    beamp=beamp/max(beamp);
    end


function H = generate_channel(N, M, L)
    % Constants (assuming half-wavelength antenna spacing)
    lambda = 0.01;           % Normalized wavelength
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
