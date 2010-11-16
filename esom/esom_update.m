function [ S ] = esom_update(S, x, varargin)
%ESOM_UPDATE Update an Evolving Self-Organizing Map
%   Updates an ESOM given in S when presented with an input (row) vector
%   x. Returns the updated map. The map S is a struct with at least the
%   following members: codebook (initially []), ids (initially []),
%   next_id (initially 1), con (initially an empty (square) sparse
%   matrix).
%
%   S = esom_update(S, x) returns the updated map S when presented an
%   input vector x.
%
%   S = esom_update(S, x, epsilon, gamma, beta, Tp) returns the updated
%   map S when presented an input vector x, given threshold (epsilon),
%   learning rate (gamma), forgetting constant (beta), and pruning
%   constant (Tp). Any number of extra parameters can be omitted. Note
%   that the syntax is likely to change in favor of a key/value-type of
%   value passing.
%  
%   The implementation is based on the paper
%   D. Deng and N. Kasabov:
%   On-line pattern analysis by evolving self-organizing maps
%   Neurocomputing, vol. 51, pp. 87-103, April 2003.
%
%   A valid initialization of the map structure is:
%     sM = struct;
%     sM.codebook = [];
%     sM.ids = [];
%     sM.next_id = 1;
%     sM.con = sparse(1000,1000); % The size 
  
%  Notes for the reader:
%    The ids vector keeps a list of connection matrix (con) indices for
%    codebook vectors. Removing a prototype means that its codebook and
%    ids entries are removed and its connections are zeroed:
%    S.con(S.ids(k),:) = 0; S.con(:,S.ids(k)) = 0;
%    S.codebook(k,:) = [];
%    S.ids(k) = [];
%
%    The new connections are assigned a weight "inf" as a placeholder.
%    The value is replaced by the real value when the network strengths
%    are updated.
%
%  TODO list:
%  - Check if modifying the sparse matrix by removing columns and rows
%    is not too expensive; if not, remove the whole inds-construct and
%    keep the codebook and connection matrix indices in sync by cutting
%    away the relevant parts of the connection matrix itself. [DO THIS]
%  - Make a separate function for pruning connections/nodes.
%  - Implement node pruning.   
%  - Make the function accept a matrix as an input to avoid the cost of
%    unnecessary function calls when data is available in advance.
%  - Add key/value option handling.
%  - Include options for variations in the algorithm:
%    - How the neighbourhood is determined
%    - Activation function
%    - Different distance metrics
%    - etc.
%  - Different distance metrics (at least cosine).
%  - Instructions how to choose the parameters?
%  - Check if there are papers on how to adaptively change the
%    parameters and implement them.
%  - Remember to make pruning connections symmetric
%  - M(:,i) is faster than M(i,:);
%  - Connection pruning might delete nodes connection to itself, which
%  leads the node not showing in the neighbourhood. Fix this properly.
%
%  Micro-optimization
%  - We don't actually need the distances pdists2 gives, the squared
%    distance would be enough.
%  - The connection matrix is symmetric, so we could only keep track of
%    the other half, but this means added logic when fiddling with the indices.
%
%  Aesthetics
%  - The way of finding three smallest values is ugly.
%  - Rename con to something more descriptive
%
%  Long-term problems
%  - Memory usage should not grow if the map size stays about constant.
%    Check if there are issues related to this in using MATLABs sparse.
%    Consider packing the sparse matrix now and then, if necessary. This
%    would also partially solve the reuse of indices issue.
%  - Theoretically, the map should be able to accept any number of
%    nodes in its lifetime. Reuse of map index values should be made
%    possible (also, the indices should not be doubles) and a maximum
%    amount of nodes should be set at some very high value
%  - Node reuse could also be solved by keeping a list of removed node
%    indices, so that small integers would be preferred when reusing
%    nodes. In all cases, remember to zero the relevant parts of the
%    connection matrix.
%
%  Copyright 2010 Ilari Nieminen <ilari.t.nieminen@iki.fi>
%  Version 0.01, 20.10.2010
  
persistent N;

if isempty(N)
   N = 0;
end
N = N + 1;

error(nargchk(2,6,nargin));

epsilon = 0.2;
gamma = 0.2;
beta = 0.8;
Tp = 50;

if (nargin > 2) 
    epsilon = varargin{1};
end
if (nargin > 3) 
    gamma = varargin{2};
end 
if (nargin > 4)     
    beta = varargin{3};
end
if (nargin > 5)
    Tp = varargin{4};
end

if (size(S.codebook,1) == 0)
    % New map
    S.codebook = [x];
    bmu = 1;
    S.con(bmu, bmu) = 1;
    return;
end

neighbours = [];
bmu = 0;
bmu2 = 0; % The second-best matching unit

dists = pdist2(S.codebook, x);

%%% Insertion of a new prototype
if (all(dists > epsilon))
    S.codebook = [S.codebook; x];
    bmu = size(S.codebook,1); % Last node in the codebook

    % Find the two closest prototypes
    [d1, n1] = min(dists);
    dists(n1) = inf; % Ruin the dists matrix
    [~, n2] = min(dists);
    dists(n1) = d1; % Fix the dists matrix

    % Connect new prototype to the two closest prototypes
    if (n1 == n2)
        % This happens when codebook has two prototypes
        neighbours = [bmu; n1];
        S.con(bmu, n1) = inf;
        S.con(n1, bmu) = inf;
    else
        neighbours = [bmu; n1; n2];
        S.con(bmu, n1) = inf;
        S.con(n1, bmu) = inf;
        
        S.con(bmu, n2) = inf;
        S.con(n2, bmu) = inf;
    end
end

%%% Find the BMU and the second-BMU if no node was inserted
if (bmu == 0)
    [d1, bmu] = min(dists);   
    dists(bmu) = inf;    
    [~, bmu2] = min(dists);
    dists(bmu) = d1; 
    % Make connection
    if (S.con(bmu, bmu2) == 0)
        S.con(bmu, bmu2) = inf;
        S.con(bmu2, bmu) = inf;
    end

    % Find the neighbouring prototypes
    neighbours = find(S.con(:, bmu));
    
end

%%% Calculate activations and update prototypes
Di = bsxfun(@minus, x, S.codebook(neighbours,:)); % (x - w_i)
DNi = sum(Di.*Di,2); % ||x - w_i||^2
A = exp(-2*DNi/epsilon^2); % a_i
a_sum = sum(A);

Delta = gamma/a_sum*bsxfun(@times,A,Di); % \delta_i
S.codebook(neighbours,:) = S.codebook(neighbours,:) + Delta;

%%% Update network connection strengths
idx = neighbours==bmu; % Index of BMU in the neighbour set
for k=1:length(neighbours)
    
    original = S.con(bmu, neighbours(k));
    activation = A(idx)*A(k); 
    value = 0;
    if original == inf
       % Initialization for new connections
       value = activation;
    else
       value = beta*original + (1-beta)*activation;
    end
    S.con(bmu, neighbours(k)) = value;
    S.con(neighbours(k), bmu) = value;    
end

%%% Connection pruning
if (mod(N,Tp) == 0)
    min_value = min(S.con(S.con>0));
    [row, col] = find(S.con == min_value, 1);
    S.con(row,col) = 0;
    S.con(col,row) = 0;
%    S.con(S.con < 2*min(S.con(S.con>0))) = 0;
end

end




    
    
    
