# Collusion Detection in Multi-Agent Games

This repository contains the MATLAB code and saved results for an eight-agent,
two-action coordination-game experiment. The statistical target is the
conditional-dependence graph \(\mathcal G^\star\) of the observed actions.

In the non-colluding benchmark, agents draw independent uniform actions in each
round. The resulting conditional-dependence graph is empty. In the colluding
regime, all eight agents participate in a hidden joint-recommendation agreement.
The recommendation device draws correlated action profiles from a pairwise
Ising law and privately sends one recommended action to each agent. The detector
observes only the resulting i.i.d. action profiles.

The planted interaction set used by the recommendation device is denoted by
\(E^{\mathrm{int}}\). For the distributions in these experiments, it equals the
edge set \(E^\star\) of the target conditional-dependence graph. The recovered
edges are statistical conditional-dependence relations; they are not observed
communication links or separate pairwise agreements.

## Repository contents

- `g2_ci/` implements conditional edge deletion with the conditional \(G^2\)
  likelihood-ratio statistic and compares it with marginal pairwise screening.
- `binary_ci/` implements the binary conditional-independence statistic and
  Poisson sampling scheme based on Algorithm 1 and Section 4 of Canonne,
  Diakonikolas, Kane, and Stewart (2018).
- Each method directory contains a `results/` folder with the saved final
  outputs used for reporting.
- `binary_ci/CALIBRATION.md` records the binary-statistic derivation,
  calibration, and population graph checks.

## Requirements

- MATLAB with `tiledlayout`, `exportgraphics`, tables, and graph plotting.
- The `g2_ci` experiment uses base MATLAB.
- The `binary_ci` experiment also requires the Statistics and Machine Learning
  Toolbox for `poissrnd`.

The full binary-CI run uses very large nominal Poisson means for its conservative
calibration rows. It remains computationally manageable because it samples the
256 profile counts directly rather than materializing every observation.

## Reproduce the saved experiments

Start MATLAB in the repository root and run:

```matlab
addpath('g2_ci');
run_g2_simulation;

addpath('binary_ci');
test_binary_ci;
run_binary_ci_simulation;
```

The first command uses seed `20260819`. The binary-CI command uses seed
`20260903` and runs 80 chain trials, 60 dense-graph trials, and 40 conservative
calibration trials by default. Each driver writes its outputs to its own
`results/` directory.

For a quick functional check with two trials per setting, use separate output
directories:

```matlab
addpath('g2_ci');
run_g2_simulation(2, 2, fullfile(tempdir, 'g2_quick_check'));

addpath('binary_ci');
test_binary_ci;
run_binary_ci_simulation(2, 2, 2, ...
    fullfile(tempdir, 'binary_ci_quick_check'));
```

## Saved outputs

The `g2_ci/results/` directory contains:

- `g2_results.csv`: recovery frequencies, precision, recall, F1 score, payoff,
  query counts, sample sizes, and seed;
- `g2_graph_recovery.pdf`: vector version of the reported figure; and
- `g2_graph_recovery.fig`: editable MATLAB figure.

The `binary_ci/results/` directory contains:

- `binary_ci_results.csv`: recovery summaries for the exploratory and
  conservatively calibrated settings;
- `population_query_margins.csv`: offline population checks and separation
  lower bounds;
- `binary_ci_raw.mat`: individual graph decisions and realized sample counts;
- `binary_ci_graph_recovery.fig` and `.png`: exact-recovery curves; and
- `binary_ci_representative_graphs.fig` and `.png`: target and representative
  estimated graphs.

The reported response tables use the `base_budget` rows from
`binary_ci_results.csv`. The `amplified_bound` rows record the separate,
conservative calibration calculation.

## Interpretation

The chain experiment uses the complete bounded-order query family with
\(d_{\max}=1\). The dense experiment uses 28 tests that condition each pair on
the other six agents; it is a full-conditioning stress test rather than the
complete 1,792-query bounded-order search.

The conditional \(G^2\) implementation uses an asymptotic chi-square
calibration. The binary-CI implementation uses Poissonized samples and the
threshold documented in `binary_ci/CALIBRATION.md`. The two procedures therefore
do not provide a comparison under matched graph-level error guarantees.
Reported Wilson intervals quantify Monte Carlo uncertainty and are not the
theorem's \(1-\delta_{\mathrm{tot}}\) graph-recovery guarantee.

## References

- L. E. Blume, “The Statistical Mechanics of Strategic Interaction,” *Games
  and Economic Behavior*, 1993. https://doi.org/10.1006/game.1993.1023
- C. L. Canonne, I. Diakonikolas, D. M. Kane, and A. Stewart, “Testing
  Conditional Independence of Discrete Distributions,” STOC 2018.
  https://doi.org/10.1145/3188745.3188756; full version:
  https://arxiv.org/abs/1711.11560v2
- S. S. Wilks, “The Large-Sample Distribution of the Likelihood Ratio for
  Testing Composite Hypotheses,” 1938.
  https://doi.org/10.1214/aoms/1177732360
- Pennsylvania State University, “Three-Way Tables: Types of Independence.”
  https://online.stat.psu.edu/stat504/Lesson05
- E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical
  Inference,” 1927. https://doi.org/10.1080/01621459.1927.10502953
