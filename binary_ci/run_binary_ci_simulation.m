function summary = run_binary_ci_simulation(sparseReplications, denseReplications, boundReplications, outputDir)
%RUN_BINARY_CI_SIMULATION Graph recovery using a binary CI statistic.
%
% Uses the same eight-agent coordination-game laws as
% ../g2_ci/run_g2_simulation.m.
% Requires Statistics and Machine Learning Toolbox for poissrnd.
%
% Two clearly separated settings:
%  base_budget: Algorithm 1 statistic/Poissonization with zeta=2 at small
%    exploratory budgets. No claim that its worst-case power bound applies.
%  amplified_bound: conservative explicit beta=2000, zeta=2, a certified
%    model-specific TV lower bound, and independent majority repetitions.
%    These numerical constants are derived in CALIBRATION.md, not given
%    numerically in the source.
%
% Large nominal sample sizes are simulated as 256 Poisson profile counts.
% This is equivalent in distribution to iid Poissonized observations, not a
% claim that collecting so many real observations is practically affordable.
%
% Example:
%   addpath('binary_ci');
%   run_binary_ci_simulation;
if nargin < 1 || isempty(sparseReplications), sparseReplications = 80; end
if nargin < 2 || isempty(denseReplications), denseReplications = 60; end
if nargin < 3 || isempty(boundReplications), boundReplications = 40; end
if nargin < 4 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
assert(all([sparseReplications denseReplications boundReplications] >= 1));
if ~exist(outputDir, 'dir'), mkdir(outputDir); end
assert(exist('poissrnd', 'file') == 2, ...
    'This experiment requires poissrnd (Statistics and Machine Learning Toolbox).');
test_binary_ci;
rng(20260903, 'twister');
timer = tic;
n = 8;
states = dec2bin(0:2^n-1, n)-'0';
allPairs = nchoosek(1:n, 2);
chain = [(1:n-1)' (2:n)'];
dense = allPairs(~(allPairs(:, 1) == 1 & allPairs(:, 2) == 2), :);
models = { ...
    struct('name', 'Independent benchmark', 'theta', 0, ...
        'pairs', zeros(0, 2), 'mode', 'order1'), ...
    struct('name', 'Chain weak', 'theta', 0.2, ...
        'pairs', chain, 'mode', 'order1'), ...
    struct('name', 'Chain strong', 'theta', 0.4, ...
        'pairs', chain, 'mode', 'order1'), ...
    struct('name', 'Dense minus one edge', 'theta', 0.12, ...
        'pairs', dense, 'mode', 'full')};
rows = struct([]);
rawResults = struct([]);
allMargins = table;
representatives = struct([]);
for modelIndex = 1:numel(models)
    model = models{modelIndex};
    p = modelLaw(states, model.pairs, model.theta);
    design = binary_ci_graph_design(states, model.mode);
    truth = ismember(design.pairs, model.pairs, 'rows');
    margins = modelMargins(p, design, truth);
    margins.model = repmat(string(model.name), height(margins), 1);
    allMargins = [allMargins; margins]; %#ok<AGROW>
    checkPopulationRecovery(design, margins, truth);
    if strcmp(model.mode, 'full')
        budgets = [1000 3000 10000 30000 100000];
        repetitions = denseReplications;
    else
        budgets = [250 500 1000 2000 5000];
        repetitions = sparseReplications;
    end
    for m = budgets
        [row, raw] = runSetting(model, p, design, truth, m, 1, ...
            repetitions, 'base_budget', NaN, NaN);
        rows = [rows; row]; %#ok<AGROW>
        rawResults = [rawResults; raw]; %#ok<AGROW>
        fprintf('%s, base m=%g: exact %d/%d\n', ...
            model.name, m, row.exact_count, repetitions);
        % Retain the first trial at the largest primary simulation budget.
        % Higher dense budgets are an additional power check, not a reason
        % to select a more successful representative.
        if (strcmp(model.mode, 'full') && m == 10000) || ...
                (~strcmp(model.mode, 'full') && m == 5000)
            representatives(modelIndex).name = model.name;
            representatives(modelIndex).truth = adjacency(n, design.pairs, truth);
            representatives(modelIndex).estimated = adjacency(n, design.pairs, raw.edgeDecisions(:, 1));
            representatives(modelIndex).baseMeanSamples = m;
        end
    end
    if any(truth)
        % Only true-edge tests require uniform rejection. For missing edges,
        % one genuine separator is enough; grey-zone nonseparator tests do
        % not spoil this sufficient event.
        epsilon = 0.5*min(margins.tv_lower_bound(margins.true_edge));
        assert(epsilon > 0);
        epsilonPrime = epsilon/2; % Source Algorithm 1, line 1.
        bins = max(design.binCount);
        m = 2000*max(sqrt(bins)/epsilonPrime^2, ...
            min(bins^(7/8)/epsilonPrime, bins^(6/7)/epsilonPrime^(8/7)));
        R = 1;
        delta = 0.05;
        while betainc(1/3, (R+1)/2, (R+1)/2) > delta/design.queryCount
            R = R+2;
        end
        [row, raw] = runSetting(model, p, design, truth, m, R, ...
            boundReplications, 'amplified_bound', epsilon, delta);
        rows = [rows; row]; %#ok<AGROW>
        rawResults = [rawResults; raw]; %#ok<AGROW>
        fprintf('%s, bound m=%.6g, R=%d: exact %d/%d\n', ...
            model.name, m, R, row.exact_count, boundReplications);
    end
end
summary = struct2table(rows);
simulationSeconds = toc(timer);
summary.simulation_seconds = repmat(simulationSeconds, height(summary), 1);
writetable(summary, fullfile(outputDir, 'binary_ci_results.csv'));
writetable(allMargins, fullfile(outputDir, 'population_query_margins.csv'));
save(fullfile(outputDir, 'binary_ci_raw.mat'), 'rawResults', 'representatives', ...
    'summary', 'allMargins', 'models', 'simulationSeconds', '-v7');
plot_binary_ci_results(outputDir);
fprintf('Simulation time excluding plots: %.3f seconds\n', simulationSeconds);
fprintf('Results saved to %s\n', outputDir);
end

function p = modelLaw(states, edges, theta)
spins = 2*states-1;
score = zeros(size(states, 1), 1);
for e = 1:size(edges, 1)
    score = score + theta*spins(:, edges(e, 1)).*spins(:, edges(e, 2));
end
p = exp(score-max(score));
p = p/sum(p);
end

function margins = modelMargins(p, design, truth)
% Offline population calculation for validation and conservative epsilon.
% The detector below receives sampled counts, not these probabilities.
joint = reshape(design.aggregate*p, 4, design.totalBins)';
determinants = joint(:, 1).*joint(:, 4)-joint(:, 2).*joint(:, 3);
lowerBound = 0.5*design.binToQuery*abs(determinants);
populationCI = design.binToQuery*abs(determinants) < 1e-13;
labels = strings(design.queryCount, 1);
for q = 1:design.queryCount
    labels(q) = string(mat2str(design.conditioning{q}));
end
pairs = design.pairs(design.queryPair, :);
margins = table(pairs(:, 1), pairs(:, 2), labels, design.binCount, ...
    truth(design.queryPair), populationCI, lowerBound, ...
    'VariableNames', {'i', 'j', 'conditioning_set', 'conditioning_bins', ...
    'true_edge', 'population_ci', 'tv_lower_bound'});
end

function checkPopulationRecovery(design, margins, truth)
assert(~any(margins.population_ci & margins.true_edge), ...
    'A true edge has a separator in the tested query family.');
separatorsPerPair = design.queryToPair*double(margins.population_ci);
assert(all(separatorsPerPair(~truth) >= 1), ...
    'Some missing edge has no separator in the tested query family.');
end

function [row, raw] = runSetting(model, p, design, truth, m, R, trials, mode, epsilon, delta)
exact = false(trials, 1);
missing = zeros(trials, 1);
extra = zeros(trials, 1);
realized = zeros(trials, 1);
edgeDecisions = false(numel(truth), trials);
thresholds = 2*sqrt(min(design.binCount, m));
assert(R*m < flintmax/16, 'Nominal counts too large for safe double arithmetic.');
for trial = 1:trials
    % Independent profile counts are exactly the Poisson splitting model.
    % Columns are independent repetition batches, shared across CI queries.
    profileCounts = poissrnd(repmat(m*p, 1, R));
    realized(trial) = sum(profileCounts(:));
    joint = permute(reshape(design.aggregate*profileCounts, ...
        4, design.totalBins, R), [2 1 3]);
    [~, phi, sizes] = binary_ci_statistic(joint);
    contributions = reshape(sizes.*phi, design.totalBins, R);
    statistics = design.binToQuery*contributions;
    accepts = sum(statistics <= thresholds, 2) > R/2;
    retained = design.queryToPair*double(accepts) == 0;
    edgeDecisions(:, trial) = retained;
    missing(trial) = sum(truth & ~retained);
    extra(trial) = sum(~truth & retained);
    exact(trial) = all(retained == truth);
end
[lo, hi] = wilson(sum(exact), trials);
row = struct('model', string(model.name), 'theta', model.theta, ...
    'mode', string(mode), 'mean_samples_per_batch', m, 'independent_batches', R, ...
    'expected_total_samples', m*R, 'mean_realized_samples', mean(realized), ...
    'conditioning_mode', string(model.mode), 'query_count', design.queryCount, ...
    'max_conditioning_bins', max(design.binCount), 'epsilon_tv', epsilon, ...
    'zeta', 2, 'beta', NaN, 'graph_failure_bound', delta, ...
    'trials', trials, 'exact_count', sum(exact), 'exact_recovery', mean(exact), ...
    'wilson95_lower', lo, 'wilson95_upper', hi, ...
    'mean_missing_edges', mean(missing), 'mean_extra_edges', mean(extra), ...
    'seed', 20260903);
if strcmp(mode, 'amplified_bound'), row.beta = 2000; end
raw = struct('model', model.name, 'mode', mode, 'meanSamples', m, ...
    'batches', R, 'edgeDecisions', edgeDecisions, 'truth', truth, ...
    'exact', exact, 'missing', missing, 'extra', extra, 'realized', realized);
end

function [lo, hi] = wilson(successes, total)
z = 1.959963984540054;
p = successes/total;
center = (p+z^2/(2*total))/(1+z^2/total);
half = z*sqrt(p*(1-p)/total+z^2/(4*total^2))/(1+z^2/total);
lo = max(0, center-half);
hi = min(1, center+half);
end

function a = adjacency(n, pairs, retained)
a = false(n);
selected = pairs(retained, :);
for e = 1:size(selected, 1)
    a(selected(e, 1), selected(e, 2)) = true;
    a(selected(e, 2), selected(e, 1)) = true;
end
end
