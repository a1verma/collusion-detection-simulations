function [statistic, phi, stratumSizes] = binary_ci_statistic(counts)
%BINARY_CI_STATISTIC Binary specialization of CDKS (2018), Algorithm 1.
% counts(z,:,r) is [n00 n01 n10 n11] for stratum z and independent batch r.
% Reference: https://arxiv.org/pdf/1711.11560v2, Sections 3 and 4.
% Phi estimates ||p_z - p_z,X p_z,Y||_2^2 = 4 det(p_z)^2 without bias.
% Do NOT truncate negative estimates: that would destroy unbiasedness.

assert(size(counts, 2) == 4, 'Each binary table must have four cells.');
assert(all(isfinite(counts(:))) && all(counts(:) >= 0) && ...
    all(counts(:) == floor(counts(:))), 'Counts must be nonnegative integers.');
counts = double(counts);
stratumSizes = sum(counts, 2);
a = counts(:, 1, :); b = counts(:, 2, :);
c = counts(:, 3, :); d = counts(:, 4, :);
ad = a .* d; bc = b .* c;
% Algebraically equal to 4*((a)_2*(d)_2+(b)_2*(c)_2-2*a*b*c*d).
% This expansion avoids subtracting two nearly equal fourth-order terms.
numerator = 4 * ((ad - bc).^2 - ad .* (a + d - 1) ...
    - bc .* (b + c - 1));
s = stratumSizes;
denominator = s .* (s - 1) .* (s - 2) .* (s - 3);
phi = zeros(size(s));
active = s >= 4;
phi(active) = numerator(active) ./ denominator(active);
statistic = reshape(sum(s .* phi, 1), 1, []);
end
