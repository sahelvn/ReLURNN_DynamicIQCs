function [g, info]=RNNgainReLU(G, N, nd)
% function [g, info]=RNNgainReLU(G, N, nd)
%
% This function analyzes the induced ell_2 norm of a ReLU recurrent neural
% network (RNN) of the form:
%   [v,e] = G [w,d]
%      w = Phi(v)   where Phi:R^nv -> R^nv is a repeated ReLU
% The repeated ReLU is assumed to apply the scalar ReLU elementwise:
%    w_k = phi( v_k ) where phi is the scalar ReLU.
% The code uses results in the reference below.
%
% Sahel Vahedi Noori, Bin Hu, Geir Dullerud, and Peter Seiler, "Stability 
% and Performance Analysis of Discrete-Time ReLU Recurrent Neural 
% Networks," submitted to the 2024 CDC.
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

% Lift system
GN = liftPlant(G,N,nd,ne);
[AN,BN,CN,DN] = ssdata(GN);
m = nv*N;

% Partion rows of CN and DN consistent with [WN;e]
CN1 = CN(1:m,:);
CN2 = CN(m+1:end,:);
DN1 = DN(1:m,:);
DN2 = DN(m+1:end,:);

% Find minimal (best) upper bound
% This uses CVX to implement the LMI condition in the reference.
cvx_begin sdp quiet
    cvx_solver mosek
    variable P(nx,nx) symmetric
    variable Q2(m,m) symmetric
    variable Q3(m,m) symmetric
    variable Qtil(m,m)    
    variable gsq
    
    minimize(gsq)    
    subject to

    % Q2 and Q3 have non-negative entries
    for i=1:m
        for j=1:m
            Q2(i,j)>=0;
            Q3(i,j)>=0;
        end
    end

    % Qtil is a Metzler matrix, i.e. all off-diag entries are non-neg.
    for i=1:m
        for j=1:m
            if i ~= j
                Qtil(i,j)>=0;
            end
        end
    end

    % Storage matrix and gain are non-negative
    P >= 0;
    gsq >= 0;
    
    % Matrix for Lyapunov function difference: V(k+1) - V(k)
    dV = [AN BN]'*P*[AN BN] - blkdiag(P,zeros(m+N*nd));
    
    % Matrix for disturbance bound term, -gsq*DN(k)' DN(k)
    Md = blkdiag(zeros(nx),zeros(m),-gsq*eye(N*nd));

    % Matrix for error output, EN(k)' EN(k)
    Me = [CN2 DN2]'*[CN2 DN2];
    
    % Matrix for quadratic contraint, [VN; WN]' Mqc [VN; WN]
    % Rfac is defined such that: [VN; WN] = Rfac*[x; WN; DN] 
    Rfac = [CN1 DN1; zeros(m,nx) eye(m) zeros(m,N*nd)];
    M = [Q2 -Qtil'-Q2'; -Qtil-Q2 Q2+Q3+Qtil+Qtil'];    
    Mqc = Rfac'*M*Rfac;

    % LMI Condition    
    tol = 0;
    dV + Md + Me + Mqc <= -tol*eye(nx+m+nd*N);
cvx_end
    
% Store data for output
info.status = cvx_status;
g = sqrt(gsq);
info.P = P;
info.Q2 = Q2;
info.Q3 = Q3;
info.Qtil = Qtil;
