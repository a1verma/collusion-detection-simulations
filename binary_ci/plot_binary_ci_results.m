function plot_binary_ci_results(outputDir)
%PLOT_BINARY_CI_RESULTS Reformat saved results without rerunning simulations.
if nargin < 1 || isempty(outputDir)
    outputDir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
saved = load(fullfile(outputDir, 'binary_ci_raw.mat'), 'summary', 'representatives');
summary = saved.summary;
representatives = saved.representatives;
f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 700]);
tiledlayout(2, 2, 'TileSpacing', 'compact');
names = unique(summary.model, 'stable');
for j = 1:numel(names)
    nexttile;
    take = summary.model == names(j) & summary.mode == "base_budget";
    t = summary(take, :);
    errorbar(t.mean_samples_per_batch, t.exact_recovery, ...
        t.exact_recovery-t.wilson95_lower, t.wilson95_upper-t.exact_recovery, ...
        '-o', 'LineWidth', 1.4, 'MarkerFaceColor', [0.2 0.4 0.8]);
    set(gca, 'XScale', 'log', 'FontSize', 11);
    ylim([-0.03 1.03]); grid on;
    title(names(j), 'Interpreter', 'none');
    xlabel('Expected observations m (one Poisson batch)');
    ylabel('Fraction recovering the entire graph');
end
sgtitle({'Binary CI statistic: small-budget exploratory runs', ...
    'Unamplified; error bars are Wilson 95% Monte Carlo intervals'}, 'FontSize', 14);
savefig(f, fullfile(outputDir, 'binary_ci_graph_recovery.fig'));
exportgraphics(f, fullfile(outputDir, 'binary_ci_graph_recovery.png'), 'Resolution', 160);
close(f);
f = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 650]);
tiledlayout(2, 3, 'TileSpacing', 'compact');
for j = 2:4
    nexttile(j-1);
    graphPanel(representatives(j).truth);
    title([representatives(j).name ': true graph'], 'Interpreter', 'none');
    nexttile(j+2);
    graphPanel(representatives(j).estimated);
    title(sprintf('First trial, m=%g', representatives(j).baseMeanSamples));
end
sgtitle({'True coordination links and representative recovered graphs', ...
    'Bottom row: first trial at the original largest budget; not selected for success'}, 'FontSize', 14);
savefig(f, fullfile(outputDir, 'binary_ci_representative_graphs.fig'));
exportgraphics(f, fullfile(outputDir, 'binary_ci_representative_graphs.png'), 'Resolution', 160);
close(f);
end

function graphPanel(a)
n = size(a, 1);
angle = 2*pi*(0:n-1)/n;
g = graph(double(a));
plot(g, 'XData', cos(angle), 'YData', sin(angle), ...
    'NodeColor', [0.15 0.35 0.75], 'EdgeColor', [0.2 0.2 0.2], ...
    'MarkerSize', 7, 'LineWidth', 1.3);
axis equal off;
xlim([-1.22 1.22]); ylim([-1.22 1.22]);
end
