function design = binary_ci_graph_design(states, mode)
%BINARY_CI_GRAPH_DESIGN Exact discrete conditioning bins for every CI query.
% The tested variables are binary. The conditioning variable is the tuple
% X_K, with 2^|K| possible values, including unobserved values.
%
% mode "order1": K is empty or a singleton (196 queries for eight agents).
% mode "full": K contains every other agent (28 queries, 64 bins each).
n = size(states, 2);
assert(all(states(:) == 0 | states(:) == 1));
pairs = nchoosek(1:n, 2);
queryPair = [];
binCount = [];
conditioning = {};
rowIndices = [];
columnIndices = [];
binQuery = [];
offset = 0;
for pairIndex = 1:size(pairs, 1)
    i = pairs(pairIndex, 1);
    j = pairs(pairIndex, 2);
    remaining = setdiff(1:n, [i j]);
    if strcmp(mode, 'order1')
        sets = [{[]} num2cell(remaining)];
    elseif strcmp(mode, 'full')
        sets = {remaining};
    else
        error('Unknown conditioning mode: %s', mode);
    end
    for setIndex = 1:numel(sets)
        K = sets{setIndex};
        q = numel(queryPair) + 1;
        bins = 2^numel(K);
        z = ones(size(states, 1), 1);
        if ~isempty(K)
            z = z + states(:, K) * (2.^(0:numel(K)-1))';
        end
        cellIndex = 1 + 2*states(:, i) + states(:, j);
        rowIndices = [rowIndices; 4*(offset+z-1)+cellIndex]; %#ok<AGROW>
        columnIndices = [columnIndices; (1:size(states, 1))']; %#ok<AGROW>
        binQuery = [binQuery; repmat(q, bins, 1)]; %#ok<AGROW>
        queryPair(q, 1) = pairIndex; %#ok<AGROW>
        binCount(q, 1) = bins; %#ok<AGROW>
        conditioning{q, 1} = K; %#ok<AGROW>
        offset = offset + bins;
    end
end
design = struct('pairs', pairs, 'queryPair', queryPair, ...
    'binCount', binCount, 'conditioning', {conditioning}, ...
    'totalBins', offset, 'queryCount', numel(queryPair));
design.aggregate = sparse(rowIndices, columnIndices, 1, ...
    4*offset, size(states, 1));
design.binToQuery = sparse(binQuery, (1:offset)', 1, ...
    design.queryCount, offset);
design.queryToPair = sparse(queryPair, (1:design.queryCount)', 1, ...
    size(pairs, 1), design.queryCount);
end
