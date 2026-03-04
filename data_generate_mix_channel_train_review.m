%% ====================================================================
% Mixed Channel Dataset Generation
% 60% Random Rician + 25% TDL-D + 15% TDL-E
%
% Purpose:
%   Generate mixed CSI dataset for NN-based channel estimation /
%   synchronization (TO + CFO) with strong generalization ability.
%
% Channel composition:
%   - Random Rician multipath (with Doppler): enhance generalization
%   - TDL-D: moderate delay spread (3GPP-like)
%   - TDL-E: long delay tail, extreme multipath
%
% Author: Haotian Liu
% Date: 19 Jan 2026
%% ====================================================================
clear all; clc;

%% ------------------ System Parameters ------------------
M_sym = 64;              % Number of OFDM symbols (time dimension)
N_c = 32;                % Number of subcarriers (frequency dimension)
N_received = 8;          % Number of receive antennas (spatial dimension)

c = 3e8;                 % Speed of light (m/s)
delta_f = 30e3;          % Subcarrier spacing (Hz)
fc = 3.5e9;              % Carrier frequency (Hz)

T = 1/delta_f + 5.975141242935e-6; 
% Effective OFDM symbol duration (including CP etc.)

a1 = 0.5;                % Normalized antenna spacing (d/lambda)

%% ------------------ Dataset Parameters ------------------
N_per_SNR = 1000;        % Number of samples per SNR point
SNR_list = -10:5:10;     % Training SNR range (dB)

%% ------------------ Channel Type Ratios ------------------
% Mixed dataset composition (per SNR)
ratio_random = 0.60;     % Random Rician (with Doppler)
ratio_TDLD   = 0.25;     % TDL-D (short delay spread)
ratio_TDLE   = 0.15;     % TDL-E (long delay tail)

N_random = round(N_per_SNR * ratio_random);
N_TDLD   = round(N_per_SNR * ratio_TDLD);
N_TDLE   = N_per_SNR - N_random - N_TDLD;

%% ------------------ Randomization Ranges ------------------
% Used only for Random multipath channel
numbers_range    = 100:500;              % Target distance (m)
numbers_velocity = 30:100;               % Radial velocity (m/s)
numbers_angle    = (30:60)*pi/180;       % AoA range (rad)

L_max = 4;                               % Max number of NLOS paths

%% ------------------ Dataset Storage ------------------
% INPUT  : noisy CSI (with TO + CFO + AWGN)
% OUTPUT : clean CSI + accumulated TO + accumulated CFO
Train_set_INPUT  = cell(N_per_SNR*length(SNR_list),1);
Train_set_OUTPUT = cell(N_per_SNR*length(SNR_list),3);

global_idx = 1;   % Global sample index across all SNRs

%% ================== Loop over SNR ==================
for SNR = SNR_list
    fprintf('==== Generating data at SNR = %d dB ====\n', SNR);

    for s = 1:N_per_SNR

        %% ================== Channel Type Selection ==================
        % Enforce fixed ratio per SNR
        if s <= N_random
            channel_type = 'Random';
        elseif s <= N_random + N_TDLD
            channel_type = 'TDL-D';
        else
            channel_type = 'TDL-E';
        end

        %% ================== LOS Component ==================
        % LOS is modeled as a rank-1 spatial steering vector
        % (no frequency selectivity, no Doppler)
        Mat_LOS = zeros(N_received,N_c,M_sym);

        theta_LOS = numbers_angle(randi(length(numbers_angle))); % AoA
        alpha_LOS = (randn + 1i*randn)/sqrt(2);                  % Complex gain

        for i = 1:N_received
            Mat_LOS(i,:,:) = ...
                alpha_LOS * exp(1i*2*pi*(i-1)*a1*sin(theta_LOS));
        end

        %% ================== NLOS Component ==================
        % Frequency-selective (and possibly time-selective) multipath
        Mat_NLOS = zeros(N_received,N_c,M_sym);

        switch channel_type

            %% -------- Random Rician Multipath (with Doppler) --------
            % Used to enhance NN generalization capability
            case 'Random'
                L = randi([1,L_max]);  % Random number of NLOS paths

                for l = 1:L
                    tau   = 2 * numbers_range(randi(end)) / c;        % Delay
                    fd    = 2 * numbers_velocity(randi(end)) * fc / c;% Doppler
                    theta = numbers_angle(randi(end));                % AoA

                    alpha = (randn + 1i*randn)/sqrt(2*L);              % Path gain

                    % Frequency response (delay)
                    k_r = exp(-1i*2*pi*(0:N_c-1)'*delta_f*tau);

                    % Time response (Doppler)
                    k_v = exp(1i*2*pi*(0:M_sym-1)*fd*T);

                    for i = 1:N_received
                        Mat_NLOS(i,:,:) = Mat_NLOS(i,:,:) + ...
                            reshape(alpha*(k_r*k_v)* ...
                            exp(1i*2*pi*(i-1)*a1*sin(theta)), ...
                            [1,N_c,M_sym]);
                    end
                end
          
            %% -------- TDL-D Channel (No Doppler) --------
            % Moderate delay spread, standard 3GPP-like profile
            case 'TDL-D'
                tau_set = [0 30 70 90 110 190 410]*1e-9;   % Delay taps
                pow_dB  = [0 -2.2 -4 -6 -8.2 -10 -15];    % Power profile

                pow = 10.^(pow_dB/10);
                pow = pow/sum(pow);  % Normalize total power

                for l = 1:length(tau_set)
                    alpha = sqrt(pow(l))*(randn+1i*randn)/sqrt(2);
                    theta = numbers_angle(randi(end));

                    k_r = exp(-1i*2*pi*(0:N_c-1)'*delta_f*tau_set(l));

                    for i = 1:N_received
                        Mat_NLOS(i,:,:) = Mat_NLOS(i,:,:) + ...
                            reshape(alpha*k_r*ones(1,M_sym)* ...
                            exp(1i*2*pi*(i-1)*a1*sin(theta)), ...
                            [1,N_c,M_sym]);
                    end
                end

            %% -------- TDL-E Channel (Long Delay Tail, No Doppler) --------
            % Extreme multipath case with long delay spread
            case 'TDL-E'
                tau_set = [0 30 150 310 370 710 1090 1730]*1e-9;
                pow_dB  = [0 -1 -2 -3 -8 -17 -20 -25];

                pow = 10.^(pow_dB/10);
                pow = pow/sum(pow);

                for l = 1:length(tau_set)
                    alpha = sqrt(pow(l))*(randn+1i*randn)/sqrt(2);
                    theta = numbers_angle(randi(end));

                    k_r = exp(-1i*2*pi*(0:N_c-1)'*delta_f*tau_set(l));

                    for i = 1:N_received
                        Mat_NLOS(i,:,:) = Mat_NLOS(i,:,:) + ...
                            reshape(alpha*k_r*ones(1,M_sym)* ...
                            exp(1i*2*pi*(i-1)*a1*sin(theta)), ...
                            [1,N_c,M_sym]);
                    end
                end
        end

        %% ================== LOS + NLOS Combination ==================
        % Random channel uses explicit Rician K-factor
        % TDL-D / TDL-E already include power profile -> no K-factor
        switch channel_type
            case 'Random'
                K_dB  = 5 + 10*rand;     % K-factor in [5,15] dB
                K_lin = 10^(K_dB/10);

                Mat_multiple1 = ...
                    sqrt(K_lin/(K_lin+1))*Mat_LOS + ...
                    sqrt(1/(K_lin+1))*Mat_NLOS;

            case {'TDL-D','TDL-E'}
                Mat_multiple1 = (Mat_LOS + Mat_NLOS) / sqrt(2);
        end

        %% ================== Generate TO & CFO ==================
        % TO: timing offset drift across OFDM symbols
        % CFO: carrier frequency offset drift
        TO_vector  = (10 + 0.01*randn(M_sym,1)) * 1e-9;
        CFO_vector = (0.001 + 0.00001*randn(M_sym,1)) * delta_f;

        TO_accu_vector  = cumsum(TO_vector);
        CFO_accu_vector = cumsum(CFO_vector);

        %% ------------------ Apply CFO ------------------
        % CFO introduces common phase rotation across subcarriers
        theta_CFO = exp(1i*2*pi*(0:M_sym-1)'*T .* CFO_accu_vector);

        CFO_tensor = zeros(N_received,N_c,M_sym);
        for m = 1:M_sym
            CFO_tensor(:,:,m) = theta_CFO(m);
        end
        Mat_CFO = Mat_multiple1 .* CFO_tensor;

        %% ------------------ Apply TO ------------------
        % TO introduces linear phase slope across subcarriers
        TO_matrix = zeros(N_c,M_sym);
        for m = 1:M_sym
            TO_matrix(:,m) = ...
                exp(-1i*2*pi*(0:N_c-1)'*delta_f*TO_accu_vector(m));
        end

        TO_tensor = zeros(N_received,N_c,M_sym);
        for n = 1:N_received
            TO_tensor(n,:,:) = TO_matrix;
        end

        Mat_final = Mat_CFO .* TO_tensor;

        %% ================== Add AWGN ==================
        H_noise = awgn(Mat_final,SNR);

        %% ================== Store Dataset ==================
        Train_set_INPUT{global_idx,1}  = H_noise;       % Network input
        Train_set_OUTPUT{global_idx,1} = Mat_multiple1;     % Clean CSI
        Train_set_OUTPUT{global_idx,2} = TO_accu_vector;% TO labels
        Train_set_OUTPUT{global_idx,3} = CFO_accu_vector;% CFO labels

        global_idx = global_idx + 1;
    end
end

%% ------------------ Save Dataset ------------------
save('INPUT_CSI_mixed_channel_train_review.mat','Train_set_INPUT','-v7.3');
save('OUTPUT_CSI_mixed_channel_train_review.mat','Train_set_OUTPUT','-v7.3');

disp('Mixed dataset generation finished.');
