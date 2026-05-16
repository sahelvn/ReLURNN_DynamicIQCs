function [g, info]=MIMOgainSlope(G, N, nd)
% function [g, info]=MIMOgainSlope(G, N, nd)
%
% This function analyzes the induced ell_2 norm of a ReLU recurrent neural
% network (RNN) of the form:
%   [v,e] = G [w,d]
%      w = Phi(v)   where Phi:R^nv -> R^nv is a repeated ReLU
% The repeated ReLU is assumed to apply the scalar ReLU elementwise:
%    w_k = phi( v_k ) where phi is the scalar ReLU.
% The code uses results in the reference below.
%
% Inputs:
% G is discrete-time plant with inputs [w;d] and outputs [v;e]
% N is the time horizon for the lifting.
% nd is the dimension of the input d
% ne is the dimention of the output e
% 
% Outputs
% g is an upper bound on the induced ell_2 norm of the ReLU RNN.
% info contains solver information.

% Get dimensions
[nOut,nIn] = size(G);
nx = size(G.A,1);
nv = nIn - nd;
ne = nOut - nv;

[A,B,C,D] = ssdata(G);
B1 = B(:,1:nv);
B2 = B(:,nv+1:end);
C1 = C(1:nv,:);
C2 = C(nv+1:end,:);
D11 = D(1:nv,1:nv);
D12 = D(1:nv,nv+1:end);
D21 = D(nv+1:end,1:nv);
D22 = D(nv+1:end,nv+1:end);

[A_psi, B1_psi, B2_psi, C_psi, D1_psi, D2_psi] = zf_fir_filter(nv, N);

% Based on Bin's paper:
A_hat = [A zeros(size(A,1),size(A_psi,2)); B1_psi*C1 A_psi];
B1_hat = [B1; B1_psi*D11+B2_psi];
B2_hat = [B2; B1_psi*D12];
C1_hat = [D1_psi*C1 C_psi];
C2_hat = [C2 zeros(size(C2,1),size(C1_hat,2)-size(C2,2))];
D11_hat = D1_psi*D11+D2_psi;
D12_hat = D1_psi*D12;
D21_hat = D21;
D22_hat = D22;

% Check realization
if false
    Psi = ss(A_psi,[B1_psi, B2_psi],C_psi,[D1_psi D2_psi],1);
    Ghat1 = blkdiag(Psi,eye(ne))*[G(1:nv,:); eye(nv) zeros(nv,nd); G(nv+1:end,:)];

    Ghat2 = ss(A_hat,[B1_hat, B2_hat],[C1_hat; C2_hat],...
        [D11_hat D12_hat; D21_hat D22_hat],1);

    norm(Ghat1-Ghat2,inf)
end


nx = size(A_hat,1);
% Find minimal (best) upper bound
% This uses CVX to implement the LMI condition in the reference.
cvx_begin sdp quiet
    cvx_solver mosek
    variable P(nx,nx) symmetric
    variable gsq
    variable m0(nv,nv)
    variable mf(nv,nv*N)
    variable mp(nv*N,nv)

    minimize(gsq)    
    subject to
    
    M0 = [m0 mf; mp zeros(nv*N,nv*N)];
    
    % All entries should be <=0 except the diagonals of m0
    % for i = 1:nv
    %     for j = 1:nv*N
    %         mf(i,j) <= 0;
    %     end
    % end
    % for i = 1:nv*N
    %     for j = 1:nv
    %         mp(i,j) <= 0;
    %     end
    % end

    mf(:) <= 0;
    mp(:) <= 0;
    for i = 1:nv
        for j = 1:nv
            if i ~= j
                m0(i,j) <= 0;
            end
        end
    end

    % Mrow = m0;
    % Mcol = m0;
    % for i=1:N
    %     idx = (1:nv) + (i-1)*nv;
    %     Mrow = [mf(:,idx) Mrow mp(idx,:)];
    %     Mcol = [mf(:,idx); Mcol; mp(idx,:)];
    % end
    % for i=1:nv
    %     sum( Mrow(i,:) ) >= 0;
    %     sum( Mcol(:,i) ) >= 0;
    % end

    Mrow = [m0 mf];
    Mcol = [m0; mp];
    for i=1:N
        idx = (1:nv) + (i-1)*nv;
        Mrow = [mp(idx,:) Mrow];
        Mcol = [mf(:,idx); Mcol];
    end
    for i=1:nv
        sum( Mrow(i,:) ) >= 0;
        sum( Mcol(:,i) ) >= 0;
    end


    % Storage matrix is positive semidefinite
    P >= 0;
    gsq >= 0;
    
    % Matrix for Lyapunov function difference: V(k+1) - V(k)
    %dV = [A_hat B1_hat B2_hat]'*P*[A_hat B1_hat B2_hat] - blkdiag(P,zeros(size(B1_hat,2)+size(B2_hat,2)));
    dV = [A_hat B1_hat B2_hat]'*P*[A_hat B1_hat B2_hat] - blkdiag(P,zeros(nv+nd));
    
    % Matrix for disturbance bound term, -gsq*DN(k)' DN(k)
    %Md = blkdiag(zeros(size(P,2)+size(B1_hat,2)),-gsq*eye(size(B2_hat,2)));
    Md = blkdiag(zeros(nx+nv),-gsq*eye(nd));

    % Matrix for error output, EN(k)' EN(k)
    Me = [C2_hat D21_hat D22_hat]'*[C2_hat D21_hat D22_hat];
    
    % Matrix for quadratic contraint, [VN; WN]' Mqc [VN; WN]
    % Rfac is defined such that: [VN; WN] = Rfac*[x; WN; DN] 
    Rfac = [C1_hat D11_hat D12_hat];
    M = [zeros(size(M0)) M0'; M0 -M0-M0'];
    Mqc = Rfac'*M*Rfac;

    % LMI Condition    
    dV + Md + Me + Mqc <= 0 * eye(size(dV,1));
cvx_end
    
% Store data for output
info.status = cvx_status;
g = sqrt(gsq);
info.P = P;
info.M0 = M0;
