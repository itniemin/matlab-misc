function [ S ] = esom_update(S, X, varargin)
%ESOM_UPDATE Update an Evolving Self-Organizing Map
%   Updates an ESOM given in S when presented with an input (row) vector
%   X. Returns the updated map. The map S is a struct with at least the
%   following members: codebook (initially []), con (initially an empty
%   (square) sparse matrix).
%
%   S = esom_update(S, X) returns the updated map S when presented an
%   input vector (or a matrix) X. In case a matrix is given, each row of
%   the matrix is considered an input vector. The performance difference is
%   small, but you can avoid writing for loops to call ESOM_UPDATE.
%
%   S = esom_update(S, X, epsilon, gamma, beta, Tp) returns the updated
%   map S when presented an input vector/matrix X, given threshold
%   (epsilon), learning rate (gamma), forgetting constant (beta), and
%   pruning constant (Tp). Any number of extra parameters can be omitted.
%   Note that the syntax is likely to change in favor of a key/value-type 
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
%     sM.con = sparse(1000,1000);
  
%  Notes for the reader:

%    The new connections are assigned a weight "inf" as a placeholder.
%    The value is replaced by the real value when the network strengths
%    are updated. The exception is formed by the diagonal entries (the
%    node's connection to itself), which are kept at infinity; the value is
%    not used anywhere.
%
%  TODO list:
%  - Make a separate function for pruning connections/nodes.
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
%
%  Micro-optimization
%  - We don't actually need the distances pdists2 gives, the squared
%    distance would be enough.
%  - The connection matrix is symmetric, so we could only keep track of
%    the other half, but this means added logic when fiddling with the
%    indices.
%
%  Aesthetics
%  - The way of finding three smallest values is ugly.
%  - Rename con to something more descriptive
%
%
%  Copyright 2010 Ilari Nieminen <ilari.t.nieminen@iki.fi>
%  Version 0.01, 20.10.2010
  
persistent N;

if isempty(N)
   N = 0;
end

error(nargchk(2,6,nargin));

% Default parameter values
epsilon = 0.2;
gamma = 0.2;
beta = 0.8;
Tp = 50;

% Handle parameters
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

first_idx = 1;

if (size(S.codebook,1) == 0)
    % New map
    N = N + 1;
    S.codebook = [X(1,:)];
    bmu = 1;
    S.con(bmu, bmu) = inf;
    % Continue from next sample if X is a matrix, otherwise we're done.
    if (size(X,1) > 1)
        first_idx = 2;
    else
        return;
    end
end

for n=first_idx:size(X,1)    
    x = X(n,:);
    N = N + 1;
    neighbours = [];
    bmu = 0; % BMU
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
        
        % Connect the new node to itself
        S.con(bmu, bmu) = inf;
        
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
        % Make connection to the closest node
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
    idx = find(neighbours==bmu); % Index of BMU in the neighbour set
    for k=1:length(neighbours)
        if k == idx
            % We don't update the node's connection to itself
            continue
        end
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
    
    %%% Connection and node pruning
    if (mod(N,Tp) == 0)
        min_value = min(S.con(S.con>0));
        [row, col] = find(S.con == min_value, 1);
        if (min_value < inf)
            S.con(row,col) = 0;
            S.con(col,row) = 0;
        end
        
        % We delete at most one connection, so there can be at most one
        % orphaned node (except when there are just two and their
        % connection has just been removed)
        
        nnzcol = arrayfun(@(k) nnz(S.con(:,k)), 1:size(S.con,1));
        if (any(nnzcol>1))
            deletable = find(nnzcol == 1, 1);
            S.codebook(deletable,:) = [];
            S.con(deletable,:) = [];
            S.con(:,deletable) = [];
        end
    end
    
end

end
 
 
 
 
 
