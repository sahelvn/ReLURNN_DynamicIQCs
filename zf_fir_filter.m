function [A, B1, B2, C, D1, D2] = zf_fir_filter(m, N)
%ZF_FIR_FILTER  Zames–Falb FIR multiplier realization
%
%   [A, B1, B2, C, D1, D2] = zf_fir_filter(m, N)
%
%   Constructs the filter used in the FIR Zames–Falb multiplier:
%
%       psi(k+1) = A*psi(k) + B1*v(k) + B2*w(k)
%       r(k)     = C*psi(k) + D1*v(k) + D2*w(k)
%
%   where:
%       v(k), w(k) ∈ R^m
%       psi(k)     ∈ R^(2*m*N)
%       r(k)       ∈ R^(2*m*(N+1))
%
%   State = [ v(k-1), ..., v(k-N),  w(k-1), ..., w(k-N) ].

% --- Shift matrix S_N and input selector b ---
if N==0
    A_single = [];
    b = zeros(0,m);
else
    S = diag(ones(N-1,1), -1);        % shift-down matrix (S_N)
    A_single = kron(S, eye(m));       % size mN × mN
    b = kron([1; zeros(N-1,1)], eye(m));   % size mN × m
end

% --- Full A matrix (block diagonal for v-register and w-register) ---
A = blkdiag(A_single, A_single);  % size 2mN × 2mN

% --- B1 injects v(k) into v-register; B2 injects w(k) into w-register ---
B1 = [b; zeros(m*N, m)];
B2 = [zeros(m*N, m); b];

% --- Output r(k) has 4 blocks:
%       r = [ v(k); v-delays; w(k); w-delays ]
%
%   Dimensions:
%       r ∈ R^(2m(N+1))
%

% Build C
C = zeros(2*m*(N+1), 2*m*N);

% Insert v-delays (rows 1+m : 1+m+mN)
C( m+1 : m+m*N , 1 : m*N ) = eye(m*N);

% Insert w-delays (bottom mN rows)
C( (m+m*N+m+1) : end ,  (m*N+1):(2*m*N) ) = eye(m*N);

% --- D1 places v(k) in the top r-block ---
D1 = zeros(2*m*(N+1), m);
D1(1:m, :) = eye(m);

% --- D2 places w(k) in its block ---
D2 = zeros(2*m*(N+1), m);
D2(m+m*N+1 : m+m*N+m, :) = eye(m);

end

