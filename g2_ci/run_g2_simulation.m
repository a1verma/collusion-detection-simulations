function summary = run_g2_simulation(sparseReplications, denseReplications, outputDir)
%RUN_G2_SIMULATION Conditional-dependence-graph recovery experiment.
%
% Eight agents play a two-action pure-coordination game on planted
% interaction links.  The independent uniform profile is the non-colluding
% benchmark.  A hidden mediator correlates recommendations by sampling an exact
% Ising law.  The script compares conditional edge deletion with marginal
% screening and reports the coordination-payoff gain.  It uses only base MATLAB
% functions and is intended for a laptop.
%
% Example:
%   addpath('g2_ci');
%   run_g2_simulation;

if nargin < 1 || isempty(sparseReplications)
    sparseReplications = 80;
end
if nargin < 2 || isempty(denseReplications)
    denseReplications = 60;
end
if nargin < 3 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

seed = 20260819;
familywiseAlpha = 0.05;
rng(seed, 'twister');
startTime = tic;

rows = struct([]);

% Sparse conditional-dependence experiment: the hidden coordination links form an
% eight-agent chain.  Every missing direct link has a separating set of size
% at most one.
n = 8;
chainPairs = [(1:n-1)' (2:n)'];
chainTruth = pairsToAdjacency(n, chainPairs);
sparseThetas = [0.20 0.40];
sparseSizes = [250 500 1000 2000 5000];

for theta = sparseThetas
    [states, probabilities] = isingDistribution(n, chainPairs, theta);
    coalitionPayoff = meanEdgeCoordinationPayoff( ...
        states, probabilities, chainPairs);
    for sampleSize = sparseSizes
        conditionalMetrics = zeros(sparseReplications, 4);
        marginalMetrics = zeros(sparseReplications, 4);
        conditionalQueries = NaN;
        marginalQueries = NaN;
        for replication = 1:sparseReplications
            data = drawSamples(states, probabilities, sampleSize);
            [estimated, conditionalQueries] = recoverBoundedOrder( ...
                data, 1, familywiseAlpha);
            conditionalMetrics(replication, :) = graphMetrics( ...
                estimated, chainTruth);

            [estimated, marginalQueries] = recoverMarginal( ...
                data, familywiseAlpha);
            marginalMetrics(replication, :) = graphMetrics( ...
                estimated, chainTruth);
        end

        newRow = summarizeMetrics('8-agent chain', theta, ...
            sampleSize, 'Conditional, order 1', sparseReplications, ...
            conditionalQueries, 2, coalitionPayoff, conditionalMetrics);
        if isempty(rows)
            rows = newRow;
        else
            rows(end + 1) = newRow; %#ok<AGROW>
        end
        rows(end + 1) = summarizeMetrics('8-agent chain', theta, ...
            sampleSize, 'Marginal screening', sparseReplications, ...
            marginalQueries, 1, coalitionPayoff, marginalMetrics); %#ok<AGROW>
    end
end

% Dense conditional-dependence stress experiment: all direct hidden coordination links are
% present except (1,2).  Testing that missing link by full conditioning uses
% the other six binary actions, so the conditioning table has 2^6 = 64 strata.
allPairs = nchoosek(1:n, 2);
densePairs = allPairs(~(allPairs(:, 1) == 1 & allPairs(:, 2) == 2), :);
denseTruth = pairsToAdjacency(n, densePairs);
denseTheta = 0.12;
denseSizes = [1000 3000 10000];
[states, probabilities] = isingDistribution(n, densePairs, denseTheta);
denseCoalitionPayoff = meanEdgeCoordinationPayoff( ...
    states, probabilities, densePairs);

for sampleSize = denseSizes
    denseMetrics = zeros(denseReplications, 4);
    denseQueries = NaN;
    for replication = 1:denseReplications
        data = drawSamples(states, probabilities, sampleSize);
        [estimated, denseQueries] = recoverFullConditioning( ...
            data, familywiseAlpha);
        denseMetrics(replication, :) = graphMetrics(estimated, denseTruth);
    end
    rows(end + 1) = summarizeMetrics('8-agent dense graph', denseTheta, ...
        sampleSize, 'Condition on remaining 6', denseReplications, ...
        denseQueries, 64, denseCoalitionPayoff, denseMetrics); %#ok<AGROW>
end

elapsedSeconds = toc(startTime);
summary = struct2table(rows);
summary.seed = repmat(seed, height(summary), 1);
summary.familywise_alpha = repmat(familywiseAlpha, height(summary), 1);
summary.total_runtime_seconds = repmat(elapsedSeconds, height(summary), 1);

csvPath = fullfile(outputDir, 'g2_results.csv');
figurePath = fullfile(outputDir, 'g2_graph_recovery.pdf');
editableFigurePath = fullfile(outputDir, 'g2_graph_recovery.fig');
writetable(summary, csvPath);
makeFigure(summary, figurePath, editableFigurePath);

fprintf('Wrote %s\n', csvPath);
fprintf('Wrote %s\n', figurePath);
fprintf('Wrote %s\n', editableFigurePath);
fprintf('Simulation time: %.1f seconds\n', elapsedSeconds);
end


function [states, probabilities] = isingDistribution(n, pairs, theta)
states = dec2bin(0:(2^n - 1), n) - '0';
spins = 2 * states - 1;
score = zeros(size(states, 1), 1);
for edge = 1:size(pairs, 1)
    i = pairs(edge, 1);
    j = pairs(edge, 2);
    score = score + theta * spins(:, i) .* spins(:, j);
end
weights = exp(score - max(score));
probabilities = weights / sum(weights);
end


function data = drawSamples(states, probabilities, sampleSize)
edges = [0; cumsum(probabilities)];
edges(end) = 1;
indices = discretize(rand(sampleSize, 1), edges);
data = states(indices, :);
end


function expectedPayoff = meanEdgeCoordinationPayoff(states, probabilities, pairs)
% Expected pure-coordination welfare per covert link: one when actions match.
payoffByProfile = zeros(size(states, 1), 1);
for edge = 1:size(pairs, 1)
    i = pairs(edge, 1);
    j = pairs(edge, 2);
    payoffByProfile = payoffByProfile + (states(:, i) == states(:, j));
end
payoffByProfile = payoffByProfile / size(pairs, 1);
expectedPayoff = sum(probabilities .* payoffByProfile);
end


function adjacency = pairsToAdjacency(n, pairs)
adjacency = false(n);
for edge = 1:size(pairs, 1)
    i = pairs(edge, 1);
    j = pairs(edge, 2);
    adjacency(i, j) = true;
    adjacency(j, i) = true;
end
end


function [estimated, queryCount] = recoverBoundedOrder(data, maxOrder, alpha)
n = size(data, 2);
pairs = nchoosek(1:n, 2);
setsPerPair = 0;
for order = 0:maxOrder
    setsPerPair = setsPerPair + nchoosek(n - 2, order);
end
queryCount = size(pairs, 1) * setsPerPair;
threshold = alpha / queryCount;
estimated = false(n);

for pair = 1:size(pairs, 1)
    i = pairs(pair, 1);
    j = pairs(pair, 2);
    remaining = setdiff(1:n, [i j]);
    removeEdge = false;
    for order = 0:maxOrder
        if order == 0
            conditioningSets = zeros(1, 0);
        else
            conditioningSets = nchoosek(remaining, order);
        end
        for setIndex = 1:size(conditioningSets, 1)
            conditioningSet = conditioningSets(setIndex, :);
            if conditionalGTest(data, i, j, conditioningSet) > threshold
                removeEdge = true;
                break;
            end
        end
        if removeEdge
            break;
        end
    end
    if ~removeEdge
        estimated(i, j) = true;
        estimated(j, i) = true;
    end
end
end


function [estimated, queryCount] = recoverMarginal(data, alpha)
n = size(data, 2);
pairs = nchoosek(1:n, 2);
queryCount = size(pairs, 1);
threshold = alpha / queryCount;
estimated = false(n);
for pair = 1:size(pairs, 1)
    i = pairs(pair, 1);
    j = pairs(pair, 2);
    if conditionalGTest(data, i, j, []) <= threshold
        estimated(i, j) = true;
        estimated(j, i) = true;
    end
end
end


function [estimated, queryCount] = recoverFullConditioning(data, alpha)
n = size(data, 2);
pairs = nchoosek(1:n, 2);
queryCount = size(pairs, 1);
threshold = alpha / queryCount;
estimated = false(n);
for pair = 1:size(pairs, 1)
    i = pairs(pair, 1);
    j = pairs(pair, 2);
    conditioningSet = setdiff(1:n, [i j]);
    if conditionalGTest(data, i, j, conditioningSet) <= threshold
        estimated(i, j) = true;
        estimated(j, i) = true;
    end
end
end


function pValue = conditionalGTest(data, i, j, conditioningSet)
if isempty(conditioningSet)
    stratum = ones(size(data, 1), 1);
    stratumCount = 1;
else
    weights = (2 .^ (0:(numel(conditioningSet) - 1)))';
    stratum = 1 + data(:, conditioningSet) * weights;
    stratumCount = 2 ^ numel(conditioningSet);
end

counts = accumarray([stratum, data(:, i) + 1, data(:, j) + 1], 1, ...
    [stratumCount, 2, 2]);
statistic = 0;
degreesOfFreedom = 0;

for value = 1:stratumCount
    contingency = squeeze(counts(value, :, :));
    rowCounts = sum(contingency, 2);
    columnCounts = sum(contingency, 1);
    total = sum(rowCounts);
    activeRows = nnz(rowCounts);
    activeColumns = nnz(columnCounts);
    if activeRows < 2 || activeColumns < 2
        continue;
    end
    expected = rowCounts * columnCounts / total;
    positive = contingency > 0;
    statistic = statistic + 2 * sum(contingency(positive) .* ...
        log(contingency(positive) ./ expected(positive)));
    degreesOfFreedom = degreesOfFreedom + ...
        (activeRows - 1) * (activeColumns - 1);
end

if degreesOfFreedom == 0
    pValue = 1;
else
    pValue = gammainc(statistic / 2, degreesOfFreedom / 2, 'upper');
end
end


function metrics = graphMetrics(estimated, truth)
estimated = triu(estimated, 1);
truth = triu(truth, 1);
truePositive = nnz(estimated & truth);
falsePositive = nnz(estimated & ~truth);
falseNegative = nnz(~estimated & truth);

if truePositive + falsePositive == 0
    precision = 0;
else
    precision = truePositive / (truePositive + falsePositive);
end
recall = truePositive / (truePositive + falseNegative);
if precision + recall == 0
    f1 = 0;
else
    f1 = 2 * precision * recall / (precision + recall);
end
exact = double(isequal(estimated, truth));
metrics = [precision recall f1 exact];
end


function row = summarizeMetrics(scenario, theta, sampleSize, method, ...
        replications, queryCount, conditioningSupport, coalitionPayoff, metrics)
if replications > 1
    standardError = std(metrics, 0, 1) / sqrt(replications);
else
    standardError = NaN(1, 4);
end
average = mean(metrics, 1);
row = struct( ...
    'scenario', scenario, ...
    'theta', theta, ...
    'sample_size', sampleSize, ...
    'method', method, ...
    'replications', replications, ...
    'query_count', queryCount, ...
    'conditioning_support', conditioningSupport, ...
    'coalition_payoff_per_edge', coalitionPayoff, ...
    'gain_over_independent', coalitionPayoff - 0.5, ...
    'precision_mean', average(1), ...
    'precision_se', standardError(1), ...
    'recall_mean', average(2), ...
    'recall_se', standardError(2), ...
    'f1_mean', average(3), ...
    'f1_se', standardError(3), ...
    'exact_mean', average(4), ...
    'exact_se', standardError(4));
end


function makeFigure(summary, figurePath, editableFigurePath)
figureHandle = figure('Visible', 'off', 'Color', 'white', ...
    'Position', [100 100 1040 720]);
layout = tiledlayout(figureHandle, 2, 2, 'Padding', 'compact', ...
    'TileSpacing', 'compact');

% Target sparse conditional-dependence graph.
nexttile(layout, 1);
chainPairs = [(1:7)' (2:8)'];
chainGraph = graph(chainPairs(:, 1), chainPairs(:, 2));
chainPlot = plot(chainGraph, 'XData', 1:8, 'YData', zeros(1, 8), ...
    'NodeLabel', string(1:8), 'NodeColor', [0.27 0.45 0.77], ...
    'EdgeColor', [0.35 0.35 0.35], 'LineWidth', 1.6, 'MarkerSize', 7);
chainPlot.NodeFontSize = 9;
axis equal;
xlim([0.4 8.6]);
ylim([-0.9 0.9]);
axis off;
title('(a) Target conditional-dependence graph: chain');
text(4.5, -0.55, '7 conditional-dependence edges', ...
    'HorizontalAlignment', 'center', 'Color', [0.2 0.2 0.2]);

% Exact recovery of the sparse graph.
nexttile(layout, 2);
hold on;
colors = [0.27 0.45 0.77; 0.77 0.35 0.07];
thetaValues = [0.20 0.40];
methods = {'Conditional, order 1', 'Marginal screening'};
markers = {'o', 's'};
lineStyles = {'-', '--'};
for thetaIndex = 1:numel(thetaValues)
    for methodIndex = 1:numel(methods)
        selected = strcmp(summary.scenario, '8-agent chain') & ...
            summary.theta == thetaValues(thetaIndex) & ...
            strcmp(summary.method, methods{methodIndex});
        part = sortrows(summary(selected, :), 'sample_size');
        label = sprintf('%s, \\theta=%.1f', ...
            strrep(methods{methodIndex}, ' screening', ''), ...
            thetaValues(thetaIndex));
        [lowerError, upperError] = wilsonErrors( ...
            part.exact_mean, part.replications);
        errorbar(part.sample_size, part.exact_mean, lowerError, upperError, ...
            'Color', colors(thetaIndex, :), ...
            'Marker', markers{methodIndex}, ...
            'LineStyle', lineStyles{methodIndex}, ...
            'LineWidth', 1.2, 'DisplayName', label);
    end
end
set(gca, 'XScale', 'log');
xlabel('sample size T');
ylabel('empirical exact-recovery frequency');
title('(b) Recovery of the chain graph');
ylim([-0.05 1.05]);
grid on;
set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
    'GridColor', [0.75 0.75 0.75]);
set(get(gca, 'Title'), 'Color', 'black');
set(get(gca, 'XLabel'), 'Color', 'black');
set(get(gca, 'YLabel'), 'Color', 'black');
legendHandle = legend('Location', 'southoutside', 'NumColumns', 2, ...
    'Interpreter', 'tex');
set(legendHandle, 'Color', 'white', 'TextColor', 'black', ...
    'EdgeColor', 'none');

% Target dense conditional-dependence graph. The dashed red segment marks the one
% absent link and is not part of the graph.
nexttile(layout, 3);
hold on;
allPairs = nchoosek(1:8, 2);
densePairs = allPairs(~(allPairs(:, 1) == 1 & allPairs(:, 2) == 2), :);
denseGraph = graph(densePairs(:, 1), densePairs(:, 2));
angles = pi / 2 - (0:7) * 2 * pi / 8;
xCoordinates = cos(angles);
yCoordinates = sin(angles);
plot(xCoordinates(1:2), yCoordinates(1:2), '--', ...
    'Color', [0.80 0.15 0.15], 'LineWidth', 1.6);
densePlot = plot(denseGraph, 'XData', xCoordinates, 'YData', yCoordinates, ...
    'NodeLabel', string(1:8), 'NodeColor', [0.44 0.68 0.28], ...
    'EdgeColor', [0.72 0.72 0.72], 'LineWidth', 0.8, 'MarkerSize', 7);
densePlot.NodeFontSize = 9;
axis equal;
xlim([-1.45 1.45]);
ylim([-1.35 1.35]);
axis off;
title('(c) Target conditional-dependence graph: dense');
text(0, -1.23, '27 links; dashed red link 1--2 is absent', ...
    'HorizontalAlignment', 'center', 'Color', [0.2 0.2 0.2]);

% Exact recovery of the dense graph.
nexttile(layout, 4);
hold on;
selected = strcmp(summary.scenario, '8-agent dense graph');
dense = sortrows(summary(selected, :), 'sample_size');
[lowerError, upperError] = wilsonErrors(dense.exact_mean, dense.replications);
errorbar(dense.sample_size, dense.exact_mean, lowerError, upperError, ...
    '-^', 'Color', [0.44 0.68 0.28], 'LineWidth', 1.2, ...
    'DisplayName', 'condition on remaining 6');
set(gca, 'XScale', 'log');
xlabel('sample size T');
ylabel('empirical exact-recovery frequency');
title('(d) Recovery of the dense graph');
ylim([-0.05 1.05]);
grid on;
set(gca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
    'GridColor', [0.75 0.75 0.75]);
set(get(gca, 'Title'), 'Color', 'black');
set(get(gca, 'XLabel'), 'Color', 'black');
set(get(gca, 'YLabel'), 'Color', 'black');
legendHandle = legend('Location', 'southoutside');
set(legendHandle, 'Color', 'white', 'TextColor', 'black', ...
    'EdgeColor', 'none');

savefig(figureHandle, editableFigurePath);
exportgraphics(figureHandle, figurePath, 'ContentType', 'vector');
close(figureHandle);
end


function [lowerError, upperError] = wilsonErrors(probability, replications)
% Asymmetric 95% Wilson intervals for a binomial recovery frequency.
z = 1.95996398454005;
denominator = 1 + z^2 ./ replications;
center = (probability + z^2 ./ (2 * replications)) ./ denominator;
halfWidth = (z ./ denominator) .* sqrt( ...
    probability .* (1 - probability) ./ replications + ...
    z^2 ./ (4 * replications.^2));
lowerBound = max(0, center - halfWidth);
upperBound = min(1, center + halfWidth);
lowerError = probability - lowerBound;
upperError = upperBound - probability;
end
