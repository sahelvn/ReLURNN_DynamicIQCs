clear all 
clc 
close all

%% ============================================================
%  Second-order discrete-time plant + NN-PI controller
% ============================================================

Ts = 0.1;

% Second-order plant
Ap = [0.88   0.08;
     -0.12   0.80];
 
Bp = [0.05;
      0.10];
 
Cp = [1 0];
 
% Controller/filter parameters
ki  = 0.20;
rho = 0.75;
 
% State:
% x = [xp; z; eta]
%
% xp  : 2 plant states
% z   : integral state
% eta : filtered error state
%
% e[k] = r[k] - y[k]
% z[k+1]   = z[k] + Ts*ki*e[k]
% eta[k+1] = rho*eta[k] + Ts*e[k]
%
% NN:
% xi = [e; z; eta]
% v  = W0*xi
% w  = phi(v)
% u  = W1*w + z
 
W0 = [ 2.0   0.0   0.0;
      -2.0   0.0   0.0;
       1.0   0.7  -0.4;
      -0.8   0.5   0.7];
 
W1 = [2.2  -1.2   1.5  -1.0];

nv = size(W0,1);
nw = nv;

nxp = size(Ap,1);
nx  = nxp + 2;
nd  = 1;
ne  = 1;

%% ============================================================
%  Build discrete-time Lurye form
% ============================================================

% x = [xp; z; eta]
%
% xp[k+1] = Ap*xp[k] + Bp*u[k]
%         = Ap*xp[k] + Bp*W1*w[k] + Bp*z[k]
%
% z[k+1] = z[k] + Ts*ki*(r[k] - Cp*xp[k])
%
% eta[k+1] = rho*eta[k] + Ts*(r[k] - Cp*xp[k])
%
% v[k] = W0*[e[k]; z[k]; eta[k]]
%      = W0*[r[k] - Cp*xp[k]; z[k]; eta[k]]
%
% e_out[k] = r[k] - y[k] = r[k] - Cp*xp[k]

A = [Ap,          Bp,       zeros(nxp,1);
     -Ts*ki*Cp,  1,        0;
     -Ts*Cp,     0,        rho];

B1 = [Bp*W1;
      zeros(1,nv);
      zeros(1,nv)];

B2 = [zeros(nxp,1);
      Ts*ki;
      Ts];

% xi = [e; z; eta]
% e = r - Cp*xp
Cxi = [-Cp,             0, 0;
       zeros(1,nxp),   1, 0;
       zeros(1,nxp),   0, 1];

Dxi = [1;
       0;
       0];

C1 = W0*Cxi;
D11 = zeros(nv,nw);
D12 = W0*Dxi;

C2  = [-Cp, 0, 0];
D21 = zeros(ne,nw);
D22 = 1;

G = ss(A, [B1, B2], ...
       [C1; C2], ...
       [D11, D12; D21, D22], Ts);

%% ============================================================
%  Sanity checks
% ============================================================

fprintf('\nSystem dimensions:\n');
fprintf('nx = %d, nv = %d, nw = %d, nd = %d, ne = %d\n', nx, nv, nw, nd, ne);

fprintf('\nOpen-loop A eigenvalue magnitudes:\n');
disp(abs(eig(A)));

fprintf('D11 norm = %.4e\n', norm(D11));
fprintf('D11 = 0 confirmed, no algebraic loop.\n');

fprintf('\nG size: %d outputs x %d inputs\n', size(G,1), size(G,2));

% disp('A = '); disp(A);
% disp('B1 = '); disp(B1);
% disp('B2 = '); disp(B2);
% disp('C1 = '); disp(C1);
% disp('D11 = '); disp(D11);
% disp('D12 = '); disp(D12);
% disp('C2 = '); disp(C2);
% disp('D21 = '); disp(D21);
% disp('D22 = '); disp(D22);

% %% ============================================================
% %  Plot NN proportional map for z = eta = 0
% % ============================================================
% 
% relu = @(x) max(0,x);
% 
% e_grid = linspace(-3,3,2000);
% z_grid = zeros(size(e_grid));
% eta_grid = zeros(size(e_grid));
% 
% xi_grid = [e_grid;
%            z_grid;
%            eta_grid];
% 
% v_grid = W0*xi_grid;
% w_grid = relu(v_grid);
% u_nn_grid = W1*w_grid;
% 
% figure;
% plot(e_grid,u_nn_grid,'LineWidth',2);
% grid on;
% xlabel('Error e');
% ylabel('NN proportional action');
% title('NN Proportional Action for z = 0, \eta = 0');

%% ============================================================
%  Run Dynamic IQCs SDP for ReLU
% ============================================================

fprintf('\nDynamic IQC ReLU:\n');
fprintf('%-6s %-12s %-12s %-12s\n', 'N', 'gamma', 'Status', 'Comp. Time');
fprintf('%s\n', repmat('-',43,1));

for N = [0,1,5,10]

    clear gR_dyn tR_dyn
    allSolved = true;

    for i = 1:10           % set i=1 for a quick run
        tic;
        [gR_dyn(i), infoR_dyn] = MIMOgainReLU(G, N, nd);
        tR_dyn(i) = toc;

        if ~strcmpi(infoR_dyn.status, 'Solved')
            allSolved = false;
        end
    end

    if allSolved
        statusStr = 'Solved';
    else
        statusStr = 'Failed';
    end

    fprintf('%-6d %-12.4f %-12s %-12.4f\n', ...
        N, mean(gR_dyn), statusStr, mean(tR_dyn));
end

%% ============================================================
%  Run Dynamic IQCs SDP for Slope
% ============================================================

fprintf('\nDynamic IQC Slope:\n');
fprintf('%-6s %-12s %-12s %-12s\n', 'N', 'gamma', 'Status', 'Comp. Time');
fprintf('%s\n', repmat('-',43,1));

for N = [0,1,5,10]

    clear gD_dyn tD_dyn
    allSolved = true;

    for i = 1:10           % set i=1 for a quick run 
        tic;
        [gD_dyn(i), infoD_dyn] = MIMOgainSlope(G, N, nd);
        tD_dyn(i) = toc;

        if ~strcmpi(infoD_dyn.status, 'Solved')
            allSolved = false;
        end
    end

    if allSolved
        statusStr = 'Solved';
    else
        statusStr = 'Failed';
    end

    fprintf('%-6d %-12.4f %-12s %-12.4f\n', ...
        N, mean(gD_dyn), statusStr, mean(tD_dyn));
end

%% ============================================================
%  Run Static QCs SDP for ReLU
% ============================================================

fprintf('\nStatic QC ReLU:\n');
fprintf('%-6s %-12s %-12s %-12s\n', 'N', 'gamma', 'Status', 'Comp. Time');
fprintf('%s\n', repmat('-',43,1));

for N = [1,2,6,11]

    clear gR_stat tR_stat
    allSolved = true;

    for i = 1:10           % set i=1 for a quick run
        tic;
        [gR_stat(i), infoR_stat] = RNNgainReLU(G, N, nd);
        tR_stat(i) = toc;

        if ~strcmpi(infoR_stat.status, 'Solved')
            allSolved = false;
        end
    end

    if allSolved
        statusStr = 'Solved';
    else
        statusStr = 'Failed';
    end

    fprintf('%-6d %-12.4f %-12s %-12.4f\n', ...
        N, mean(gR_stat), statusStr, mean(tR_stat));
end

%% ============================================================
%  Run Static QCs SDP for Slope
% ============================================================

fprintf('\nStatic QC Slope:\n');
fprintf('%-6s %-12s %-12s %-12s\n', 'N', 'gamma', 'Status', 'Comp. Time');
fprintf('%s\n', repmat('-',43,1));

for N = [1,2,6,11]

    clear gD_stat tD_stat
    allSolved = true;

    for i = 1:10           % set i=1 for a quick run
        tic;
        [gD_stat(i), infoD_stat] = RNNgainDH(G, N, nd);
        tD_stat(i) = toc;

        if ~strcmpi(infoD_stat.status, 'Solved')
            allSolved = false;
        end
    end

    if allSolved
        statusStr = 'Solved';
    else
        statusStr = 'Failed';
    end

    fprintf('%-6d %-12.4f %-12s %-12.4f\n', ...
        N, mean(gD_stat), statusStr, mean(tD_stat));
end