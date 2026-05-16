function [g, info]=RNNgainDH(G, N, nd)
% function [g, info]=RNNgainDH(G, N, nd)
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
    variable Q0(m,m)
    variable gsq
    
    minimize(gsq)    
    subject to
    
    % Q0 is doubly hyperdominant:  Off diagonal entries are non-positive
    % and row/column sums are non-negative.
    for i=1:m
        for j=1:m
            if i ~= j
                Q0(i,j)<=0;
            end
        end

        % i^th row sum is >=0
        Q0(i,:)*ones(m,1)>=0;

        % i^th row column is >=0
        ones(1,m)*Q0(:,i)>=0;
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
    M = [zeros(m) Q0'; Q0 -Q0-Q0'];
    Mqc = Rfac'*M*Rfac;

    % LMI Condition    
    tol = 0;
    dV + Md + Me + Mqc <= -tol*eye(nx+m+nd*N);
cvx_end
    
% Store data for output
info.status = cvx_status;
g = sqrt(gsq);
info.P = P;
info.Q0 = Q0;
