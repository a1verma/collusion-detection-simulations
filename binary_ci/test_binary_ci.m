function results = test_binary_ci
%TEST_BINARY_CI Deterministic tests, not empirical theorem validation.
distributions = [0.25 0.25 0.25 0.25; ...
    0.4 0.1 0.1 0.4; 0.12 0.18 0.28 0.42; ...
    0.1 0.2 0.3 0.4; 0.5 0 0 0.5];
sampleSizes = [4 5 8 12];
largestMeanError = 0;
largestVarianceError = 0;
for s = sampleSizes
    tables = zeros(nchoosek(s + 3, 3), 4);
    index = 0;
    for a = 0:s
        for b = 0:(s-a)
            for c = 0:(s-a-b)
                index = index + 1;
                tables(index, :) = [a b c s-a-b-c];
            end
        end
    end
    [~, estimates] = binary_ci_statistic(tables);
    estimates = estimates(:);
    for pIndex = 1:size(distributions, 1)
        p = distributions(pIndex, :);
        masses = exp(gammaln(s+1) - sum(gammaln(tables+1), 2));
        for cellIndex = 1:4
            masses = masses .* p(cellIndex).^tables(:, cellIndex);
        end
        assert(abs(sum(masses)-1) < 1e-12);
        target = 4*(p(1)*p(4)-p(2)*p(3))^2;
        meanEstimate = sum(masses .* estimates);
        largestMeanError = max(largestMeanError, abs(meanEstimate-target));
        assert(abs(meanEstimate-target) < 2e-13);
        if pIndex == 1 || pIndex == 3
            rowMass = p(1)+p(2);
            columnMass = p(1)+p(3);
            t = rowMass*(1-rowMass)*columnMass*(1-columnMass);
            variance = sum(masses .* (estimates-target).^2);
            expectedVariance = 32*t^2/(s*(s-3));
            largestVarianceError = max(largestVarianceError, ...
                abs(variance-expectedVariance));
            assert(abs(variance-expectedVariance) < 2e-13);
        end
    end
end
[A, phi] = binary_ci_statistic([1 1 1 1; 1 0 1 0]);
assert(abs(phi(1)+1/3) < 1e-14 && phi(2) == 0);
assert(abs(A+4/3) < 1e-14);
assert(binary_ci_test([1 1 1 1], 4, 2));
assert(~binary_ci_test([100 0 0 100], 200, 2));
reference = binary_ci_statistic([9 5 3 7; 6 10 4 2]);
assert(abs(reference-binary_ci_statistic([3 7 9 5; 4 2 6 10])) < 1e-13);
assert(abs(reference-binary_ci_statistic([9 3 5 7; 6 4 10 2])) < 1e-13);
% Independently reconstruct every discrete table to check sparse aggregation.
states = dec2bin(0:255, 8)-'0';
profileCounts = mod((1:256)', 13);
for mode = {'order1', 'full'}
    design = binary_ci_graph_design(states, mode{1});
    tables = reshape(design.aggregate*profileCounts, 4, design.totalBins)';
    [~, phi, sizes] = binary_ci_statistic(tables);
    groupedStatistics = design.binToQuery*(sizes.*phi);
    offset = 0;
    for q = 1:design.queryCount
        ij = design.pairs(design.queryPair(q), :);
        K = design.conditioning{q};
        direct = zeros(design.binCount(q), 4);
        for state = 1:size(states, 1)
            z = 1;
            for k = 1:numel(K)
                z = z + states(state, K(k))*2^(k-1);
            end
            cellIndex = 1 + 2*states(state, ij(1)) + states(state, ij(2));
            direct(z, cellIndex) = direct(z, cellIndex)+profileCounts(state);
        end
        assert(isequal(direct, tables(offset+(1:design.binCount(q)), :)));
        assert(abs(binary_ci_statistic(direct)-groupedStatistics(q)) < 1e-12);
        offset = offset+design.binCount(q);
    end
    if strcmp(mode{1}, 'order1')
        assert(design.queryCount == 196 && max(design.binCount) == 2);
    else
        assert(design.queryCount == 28 && all(design.binCount == 64));
    end
    % Check the additional repetition dimension and majority graph decisions.
    batches = [profileCounts, 3*profileCounts, flipud(profileCounts)];
    stacked = permute(reshape(design.aggregate*batches, ...
        4, design.totalBins, 3), [2 1 3]);
    [~, batchedPhi, batchedSizes] = binary_ci_statistic(stacked);
    batchedA = design.binToQuery*reshape(batchedSizes.*batchedPhi, design.totalBins, 3);
    slowAccepts = false(design.queryCount, 3);
    offset = 0;
    for q = 1:design.queryCount
        for batch = 1:3
            oneTable = stacked(offset+(1:design.binCount(q)), :, batch);
            [slowAccepts(q, batch), slowA] = binary_ci_test(oneTable, 500, 2);
            assert(abs(slowA-batchedA(q, batch)) < 1e-12);
        end
        offset = offset+design.binCount(q);
    end
    fastAccepts = batchedA <= 2*sqrt(min(design.binCount, 500));
    assert(isequal(slowAccepts, fastAccepts));
    fastGraph = design.queryToPair*double(sum(fastAccepts, 2) >= 2) == 0;
    slowGraph = true(size(design.pairs, 1), 1);
    for q = 1:design.queryCount
        if sum(slowAccepts(q, :)) >= 2
            slowGraph(design.queryPair(q)) = false;
        end
    end
    assert(isequal(fastGraph, slowGraph));
end
% X1=X2=X3: dependence marginally, independence given X3 (degenerate bins).
assert(binary_ci_statistic([100 0 0 100]) > 0);
assert(binary_ci_statistic([100 0 0 0; 0 0 0 100]) == 0);
gamma = 1-5/(2*exp(1));
muConstant = 2000*gamma/8;
alternativeBound = (5.055*muConstant+16) / ...
    (5.055*muConstant+16+(muConstant-2)^2);
assert(alternativeBound < 1/3);
results = struct('passed', true, 'largest_mean_error', largestMeanError, ...
    'largest_null_variance_error', largestVarianceError, ...
    'base_null_error_bound', 1/9, ...
    'base_alternative_error_bound', alternativeBound);
disp(results);
end
