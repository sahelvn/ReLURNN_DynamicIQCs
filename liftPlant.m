function GN = liftPlant(G,N,nd,ne)
% function GN = liftPlant(G,N,nd,ne)
%
% This function constructs a lifted representation for a discrete-time
% system G with inputs (w,d) and outputs (v,e).
%
% Inputs:
% G is discrete-time plant with inputs [w;d] and outputs [v;e]
% N is the time horizon for the lifting.
% nd is the dimension of the input d
% ne is the dimension of the output e
%
% Output:
% GN is the lifted plant with [VN; EN] = GN [WN; DN] where 
%    (WN, DN, VN, EN) are the signals lifted on an N-step horizon, e.g.
%    WN(k) = [w(k);  w(k-1); ...; w(k-N+1)]

% Error check
if floor(N)~=ceil(N) || N<=0
    error('N must be an integer >=1');
end

% Get plant data
nv = size(G,1)-ne;
nw = size(G,2)-nd;

[A,B,C,D] = ssdata(G);
B1 = B(:,1:nw);
B2 = B(:,nw+1:end);
C1 = C(1:nv,:);
C2 = C(nv+1:end,:);
D11 = D(1:nv,1:nw);
D12 = D(1:nv,nw+1:end);
D21 = D(nv+1:end,1:nw);
D22 = D(nv+1:end,nw+1:end);

% Store powers of A
nx = size(A,1);
Apow = cell(N,1);
Apow{1} = A;
for i=2:N
    Apow{i} = A*Apow{i-1};
end
AN = Apow{N};

% Build up lifted system
for i=1:N
    if i==1
        B1N = B1;
        B2N = B2;
        C1N = C1;
        C2N = C2;
        D11N = D11;
        D12N = D12;
        D21N = D21;
        D22N = D22;
    else
        B1N = [Apow{i-1}*B1  B1N];
        B2N = [Apow{i-1}*B2  B2N];
        C1N = [C1N; C1*Apow{i-1}];
        C2N = [C2N; C2*Apow{i-1}];        
        if i==2
            D11N = [D11N zeros(nv,nw); C1*B1 D11N];
            D12N = [D12N zeros(nv,nd); C1*B2 D12N];
            D21N = [D21N zeros(ne,nw); C2*B1 D21N];
            D22N = [D22N zeros(ne,nd); C2*B2 D22N];
        else
            D11N = [D11N zeros((i-1)*nv,nw); ...
                C1*Apow{i-2}*B1 D11N(end-nv+1:end,:)];
            D12N = [D12N zeros((i-1)*nv,nd); ...
                C1*Apow{i-2}*B2 D12N(end-nv+1:end,:)];            
            D21N = [D21N zeros((i-1)*ne,nw); ...
                C2*Apow{i-2}*B1 D21N(end-ne+1:end,:)];            
            D22N = [D22N zeros((i-1)*ne,nd); ...
                C2*Apow{i-2}*B2 D22N(end-ne+1:end,:)];
        end
    end
end

BN = [B1N B2N];
CN = [C1N; C2N];
DN = [D11N D12N; D21N D22N];
GN = ss(AN,BN,CN,DN,G.Ts);

