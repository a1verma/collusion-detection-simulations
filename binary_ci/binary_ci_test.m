function [acceptCI, statistic, threshold] = binary_ci_test(counts, meanSamples, zeta)
%BINARY_CI_TEST Algorithm 1 applied to Poissonized binary tables.
% meanSamples is the Poisson MEAN, not the realized sample count.
% Small statistic means accept CI; this follows Algorithm 1 lines 13--16.
% The paragraph immediately before that algorithm reverses this direction.
if nargin < 3
    zeta = 2;
end
assert(isscalar(meanSamples) && meanSamples > 0);
assert(isscalar(zeta) && zeta > 0);
statistic = binary_ci_statistic(counts);
threshold = zeta * sqrt(min(size(counts, 1), meanSamples));
acceptCI = statistic <= threshold;
end
