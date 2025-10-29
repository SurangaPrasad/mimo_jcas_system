clear; close all; clc; rng(0);

%% ---------------- Simulation Parameters ----------------
N = 64;  
M = 4;  
K = 4;
P = 3; 
Ncl = 4;  % Number of clusters
Nray = 10;  % Number of rays per cluster
theta_desired = [-60 0 60];  
delta_theta = 5;
omega = 0.3;  
sigma_n2 = 1;
Imax = 120;   
mu_const = 0.01;  
lambda_const = 0.01;
eta = 1/N;    
J_values = [1 10 20];
SNR_dB = 0:2:12;
PBS_from_snr = @(snr_db) sigma_n2*10.^(snr_db/10);
T = 181;  
theta_grid = linspace(-90,90,T).';
num_realizations = 100;   % Increased from 10 to 100

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
    if exist('cvx_begin','file')
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
        cvx_end
        if ~strcmp(cvx_status, 'Solved') && ~strcmp(cvx_status, 'Inaccurate/Solved')
            warning('CVX did not converge, using identity');
            Psi = (PBS/N)*eye(N);
        end
    else
        warning('CVX unavailable — using Ψ = (PBS/N) I_N');
        Psi = (PBS/N)*eye(N);
    end

    % --- Averages over channel realizations ---
    rPGA=0; rU1=0; rU10=0; rU20=0; rZF=0;
    mPGA=0; mU1=0; mU10=0; mU20=0; mZF=0;

    for rr = 1:num_realizations
        % Generate mmWave channel using Saleh-Valenzuela model
        H = generate_mmwave_channel(N, K, Ncl, Nray);

        % Initialization (Eq. 17) - corrected
        if M > K
            steering_des = exp(1j*pi*(0:N-1)'.*sin(deg2rad(theta_desired(1:M-K))));
            G = [H', steering_des];
        else
            G = H';
        end
        A0 = exp(-1j*angle(G(:,1:M)));
        
        % Digital precoder initialization
        XZF = pinv(H);  % Note: H is K×N, so H' is N×K
        D0 = pinv(A0)*XZF;
        D0 = D0*sqrt(PBS)/norm(A0*D0,'fro');

        % ----- Conventional PGA (J=1) -----
        [A_pga,D_pga] = run_PGA(A0,D0,H,Psi,PBS,omega,eta,mu_const,lambda_const,Imax,1,K);
        [rTmp,mTmp,bTmp] = metrics(A_pga,D_pga,H,Psi,Bd,Abar_grid,sigma_n2);
        rPGA=rPGA+rTmp; 
        mPGA=mPGA+mTmp;
        if sIdx==numSNR && rr==1, beamSnap.PGA=bTmp; end

        % ----- UPGANet J=1,10,20 -----
        for Jv = J_values
            [A_u,D_u] = run_PGA(A0,D0,H,Psi,PBS,omega,eta,mu_const,lambda_const,Imax,Jv,K);
            [rTmp,mTmp,bTmp] = metrics(A_u,D_u,H,Psi,Bd,Abar_grid,sigma_n2);
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

    % Averages
    R_PGA(sIdx)=rPGA/num_realizations;  MSE_PGA(sIdx)=mPGA/num_realizations;
    R_UPG1(sIdx)=rU1/num_realizations;  MSE_UPG1(sIdx)=mU1/num_realizations;
    R_UPG10(sIdx)=rU10/num_realizations;MSE_UPG10(sIdx)=mU10/num_realizations;
    R_UPG20(sIdx)=rU20/num_realizations;MSE_UPG20(sIdx)=mU20/num_realizations;
    R_ZF(sIdx)=rZF/num_realizations;    MSE_ZF(sIdx)=mZF/num_realizations;
end

%% ---------------- Plot Figure 3 ----------------
figure('Color','w','Position',[100 100 1200 400]);

subplot(1,3,1); hold on; grid on;
plot(SNR_dB,R_UPG1,'-o','LineWidth',1.5,'MarkerSize',6); 
plot(SNR_dB,R_UPG10,'-s','LineWidth',1.5,'MarkerSize',6); 
plot(SNR_dB,R_UPG20,'-^','LineWidth',1.5,'MarkerSize',6);
plot(SNR_dB,R_PGA,'--','LineWidth',1.5);  
plot(SNR_dB,R_ZF,':','LineWidth',2);
xlabel('SNR [dB]','FontSize',11); 
ylabel('R [bits/s/Hz]','FontSize',11);
legend('UPGANet (J=1)','UPGANet (J=10)','UPGANet (J=20)',...
       'Conventional PGA','ZF (digital, comm. only)','Location','best','FontSize',9);
title('(a) Communications sum rate','FontSize',11);
xlim([0 12]); ylim([0 30]);

subplot(1,3,2); hold on; grid on;
plot(SNR_dB,MSE_UPG1,'-o','LineWidth',1.5,'MarkerSize',6); 
plot(SNR_dB,MSE_UPG10,'-s','LineWidth',1.5,'MarkerSize',6); 
plot(SNR_dB,MSE_UPG20,'-^','LineWidth',1.5,'MarkerSize',6);
plot(SNR_dB,MSE_PGA,'--','LineWidth',1.5); 
plot(SNR_dB,MSE_ZF,':','LineWidth',2);
xlabel('SNR [dB]','FontSize',11); 
ylabel('Avg beampattern MSE [dB]','FontSize',11);
legend('UPGANet (J=1)','UPGANet (J=10)','UPGANet (J=20)',...
       'Conventional PGA','ZF (digital, comm. only)','Location','best','FontSize',9);
title('(b) Average beampattern MSE','FontSize',11);
xlim([0 12]);

subplot(1,3,3); hold on; grid on; box on;
% Plot benchmark beampattern (normalized to 0 dB at peaks)
plot(theta_grid,10*log10(Bd+1e-12),'k-','LineWidth',2);
fields={'UPG20','UPG10','UPG1','PGA','ZF'};
lineStyles = {'-','-','-','--',':'};
for idx=1:length(fields)
    f = fields{idx};
    if isfield(beamSnap,f)
        plot(theta_grid,10*log10(beamSnap.(f)+1e-12),...
             lineStyles{idx},'LineWidth',1.5);
    end
end
xlabel('Angle (°)','FontSize',11); 
ylabel('Normalized sensing beampattern [dB]','FontSize',11);
legend('Benchmark beampattern','UPGANet (J=20)','UPGANet (J=10)',...
       'UPGANet (J=1)','Conventional PGA','ZF (digital, comm. only)',...
       'Location','best','FontSize',9);
title('(c) Sensing beampattern','FontSize',11);
xlim([-90 90]); ylim([-40 5]);

sgtitle('Figure 3 — UPGANet vs PGA vs ZF','FontSize',12,'FontWeight','bold');

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

function [Rval,MSEdB,beamp] = metrics(A,D,H,Psi,Bd,Abar,sigma2)
    PsiAD=A*D*D'*A'; 
    T=size(Abar,2);
    Rval=0; K=size(H,1);
    for k=1:K
        hk=H(k,:)';
        num=abs(hk'*A*D(:,k))^2;
        den=sigma2+sum(abs(hk'*A*D).^2)-num;
        Rval=Rval+log2(1+num/den);
    end
    beamp=abs(diag(Abar'*PsiAD*Abar));
    MSEdB=10*log10(mean(abs(Bd-beamp).^2)+1e-12);
    beamp=beamp/max(beamp+1e-12);
end

function H = generate_mmwave_channel(N, K, Ncl, Nray)
    % Extended Saleh-Valenzuela model for mmWave channels
    % Following the paper's references [16], [26], [27]
    % H is K×N where H = [h1; h2; ...; hK]
    
    H = zeros(K, N);
    
    for k = 1:K
        % Path loss and shadow fading (simplified)
        PL = 1;  % Normalized path loss
        
        hk = zeros(N, 1);  % Column vector for user k
        
        for i = 1:Ncl
            % Cluster angles
            phi_i = (rand - 0.5) * pi;  % Cluster AoD
            
            % Cluster gain
            alpha_i = (randn + 1j*randn)/sqrt(2) * sqrt(PL);
            
            for l = 1:Nray
                % Ray angles with angle spread
                sigma_phi = 10 * pi/180;  % 10 degree angle spread
                phi_il = phi_i + sigma_phi * randn;
                
                % Ray gain
                beta_il = (randn + 1j*randn)/sqrt(2);
                
                % Steering vector (N×1)
                a_t = exp(1j*pi*(0:N-1)'*sin(phi_il)) / sqrt(N);
                
                % Add ray contribution
                hk = hk + alpha_i * beta_il * a_t;
            end
        end
        
        % Normalize by number of paths and store as row in H
        H(k,:) = (hk * sqrt(N / (Ncl * Nray))).';
    end
end