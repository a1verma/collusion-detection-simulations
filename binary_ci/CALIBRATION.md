# Explicit calibration for the binary-CI experiment

This note documents the estimator, sampling scheme, and calibration used by
the binary-CI implementation. The construction specializes Algorithm 1 of
[Canonne, Diakonikolas, Kane and Stewart (2018)](https://arxiv.org/pdf/1711.11560v2).
The numerical choice \(\beta=2000,\zeta=2\) is a conservative derivation for
this implementation; the authors do not supply those numerical constants.
These bounds are sufficient, not necessary sample requirements.

## 1. Binary statistic and an explicit variance bound

Write a conditional probability table as \(p=(a,b,c,d)\), in cell order
\((00,01,10,11)\), and let \(D=ad-bc\). Each entry of
\(p-p_X\otimes p_Y\) is either \(D\) or \(-D\), so
\[
Q(p)=\|p-p_X\otimes p_Y\|_2^2=4D^2.
\]
For a bin of size \(s\ge4\), with counts \(N=(N_a,N_b,N_c,N_d)\), its
unbiased estimator is
\[
\Phi(N)=\frac{4\{(N_a)_2(N_d)_2+(N_b)_2(N_c)_2
                 -2N_aN_bN_cN_d\}}{(s)_4}.
\]
Here \((u)_r=u(u-1)\cdots(u-r+1)\). The multinomial factorial-moment identity
\[
E\!\left[\prod_i(N_i)_{r_i}\mid s\right]
 =(s)_{\sum_i r_i}\prod_i p_i^{r_i}
\]
proves unbiasedness term by term. Negative estimates are retained.

Equivalently, \(\Phi\) is the average of a symmetric kernel \(h\) over all
four-element subsets of the observations. The kernel is \(2/3\) on count
patterns \((2,0,0,2)\) and \((0,2,2,0)\), \(-1/3\) on \((1,1,1,1)\), and
zero otherwise. This representation is for analysis: the code evaluates
the closed-form count formula, not all subsets.

Let \(V_j=\operatorname{Var}(E[h\mid W_1,\ldots,W_j])\), for iid draws
\(W_i\sim p\). Two independent, uniformly selected four-subsets overlap in
\(J\) indices. Conditional independence of their nonshared observations
gives
\[
\operatorname{Var}(\Phi\mid s)
 =\sum_{j=1}^4
 \frac{\binom4j\binom{s-4}{4-j}}{\binom s4}V_j.
\]
Impossible binomial coefficients are zero. The one-draw conditional
expectation is
\[
E[h\mid W_1=\cdot]=2D(d,-c,-b,a).
\]
Consequently \(V_1=Q S-Q^2\), where
\[
S=ad(a+d)+bc(b+c)
 \le\frac{(a+d)^3+(b+c)^3}{4}\le\frac14.
\]
Thus \(V_1\le Q/4\). Also
\[
E[h^2]=\frac83\{(ad)^2+(bc)^2+abcd\}
 \le\frac83(ad+bc)^2\le\frac16,
\]
since \(ad+bc\le1/4\). Conditional expectation cannot increase variance,
so \(V_j\le1/6\) for the other projections.

The overlap satisfies \(P(J=1)\le E[J]=16/s\) and
\[
P(J\ge2)\le E\binom J2=\frac{72}{s(s-1)}.
\]
Combining these bounds gives
\[
\boxed{\operatorname{Var}(\Phi\mid s)
 \le \frac{4Q}{s}+\frac{12}{s(s-1)}.}
\]

## 2. Combining bins, including random bin sizes

Let \(b=|\mathcal Z|\), \(N_z\sim\operatorname{Poisson}(mP_Z(z))\),
\(A=\sum_z N_z\Phi_z\mathbf1_{\{N_z\ge4\}}\),
\(\mu=E[A]\), and \(k=\min(b,m)\).
The Poisson bin counts are independent. The conditional variance bound gives
\[
E[\operatorname{Var}(A\mid(N_z)_z)]
 \le4\mu+16\sum_zP(N_z\ge4),
\]
because \(12s/(s-1)\le16\) for \(s\ge4\).
Claim 2.1 of the cited paper supplies
\(\operatorname{Var}(N\mathbf1_{\{N\ge4\}})
\le4.22E[N\mathbf1_{\{N\ge4\}}]\).
Since \(0\le Q_z\le1/4\), the variance of the conditional mean is at most
\(1.055\mu\). Finally,
\(\sum_zP(N_z\ge4)\le\min(b,m)=k\), using
\(P(N_z\ge4)\le E[N_z]\). Hence
\[
\boxed{\operatorname{Var}(A)\le5.055\mu+16k.}
\]

There is a sharper null bound. Under independence within a bin, let
\(r=P(X=0)\), \(c=P(Y=0)\), and \(t=r(1-r)c(1-c)\le1/16\).
For the kernel above, direct conditional expectations give
\[
V_1=0,\quad V_2=\frac49t^2,\quad V_3=2t^2,\quad V_4=8t^2.
\]
Substitution into the overlap formula simplifies to
\[
\operatorname{Var}(\Phi\mid s)
=\frac{32t^2}{s(s-3)}\le\frac1{8s(s-3)}.
\]
Since \(s/(s-3)\le4\), summing over bins gives
\[
\mu=0,\qquad \operatorname{Var}(A)\le k/2.
\]
This includes degenerate independent distributions, for which \(t=0\).

## 3. A concrete threshold and sufficient sample budget

Set \(\tau=2\sqrt{k}\). The one-sided variance inequality
\[
P(W-EW\ge u)\le
\frac{\operatorname{Var}(W)}{\operatorname{Var}(W)+u^2}
\]
therefore bounds the null rejection probability by \(1/9\).

For the alternative, take \(0<\varepsilon\le1\) and use total variation
\(d_{\mathrm{TV}}(P,Q)=\frac12\|P-Q\|_1\). Set \(\varepsilon'=\varepsilon/2\)
for binary tested variables and choose
\[
m\ge\beta\max\left\{
\frac{\sqrt b}{(\varepsilon')^2},
\min\left(\frac{b^{7/8}}{\varepsilon'},
          \frac{b^{6/7}}{(\varepsilon')^{8/7}}\right)
\right\}.
\]
Proposition 3.1 of the source yields
\(\mu\ge(\beta\gamma/8)\sqrt{k}\) when
\(d_{\mathrm{TV}}(P,\mathcal P_{\mathrm{CI}})>\varepsilon\), with
\(\gamma=1-5/(2e)\).
Choose \(\beta=2000\); then \(a=\beta\gamma/8=20.075349\ldots\).
The variance inequality now gives
\[
P(A\le2\sqrt{k})
\le\frac{5.055\mu+16k}
         {5.055\mu+16k+(\mu-2\sqrt{k})^2}.
\]
The right-hand side decreases for \(\mu>2\sqrt{k}\).
At \(\mu=a\sqrt{k}\), division by \(k\) and \(k\ge1\) bound it by
\[
\frac{5.055a+16}{5.055a+16+(a-2)^2}
=0.26447798\ldots<\frac13.
\]
Thus both error types are at most \(1/3\) at this sufficient budget.
At exploratory budgets below it, this argument does not certify power.

Implementation follows Algorithm 1's decision \(A\le\tau\) for acceptance.
The preceding prose in the source reverses that inequality; the displayed
algorithm and its proof determine the implemented direction. The threshold
constant must satisfy the null bound and then the sample constant must be
large enough for power; blindly taking the threshold arbitrarily small is
not justified.

## 4. A safe model-specific TV lower bound

For any joint distribution \(P\), form the *unnormalized* tables
\(B_z(x,y)=P(X=x,Y=y,Z=z)\), and define
\[
L(P)=\frac12\sum_z|\det B_z|.
\]
For every \(Q\in\mathcal P_{\mathrm{CI}}\), its corresponding tables \(C_z\)
have zero determinant, including zero-mass bins. On the segment between
two nonnegative subprobability tables, every coordinate of the gradient of
the determinant has magnitude at most one. Integrating the gradient gives
\[
|\det B_z-\det C_z|\le\|B_z-C_z\|_1.
\]
After summing and dividing by two,
\[
\boxed{d_{\mathrm{TV}}(P,\mathcal P_{\mathrm{CI}})\ge L(P).}
\]
The minimization permits changing both the \(Z\)-marginal and all conditional
marginals. This is not distance to a fixed uniform or marginal-product law.
The bound is conservative; no exact-distance claim is made.

For the experiment, let \(\mathcal Q_E\) be all tested queries whose pair is
a true graph edge, and take
\[
\varepsilon=\frac12\min_{q\in\mathcal Q_E}L(P_q).
\]
Then every such query is strictly more than \(\varepsilon\) from CI.
Population probabilities are used only for this offline choice and truth
validation; test statistics use sampled counts. This is an oracle-assisted
calibration of a known simulation model, not an estimator of an unknown
\(c^*\) from observed data.

## 5. From correct CI decisions to the whole graph

The chain and dense experiment have the following exact population
properties. In the positive pairwise Ising law, full conditioning gives
odds ratio \(e^{4\theta_{ij}}\); it is one exactly when the edge is absent.
For a chain, conditioning on an interior vertex of the path separates any
nonadjacent endpoints. For a true chain edge, removing that edge splits
the graph into two components; summing or conditioning on other vertices
contributes separate endpoint factors \(f(x_i)g(x_j)\), leaving odds ratio
\(e^{4\theta}>1\). Thus no tested conditioning set separates a true edge.
The numerical population checks in the code validate this structure and
the indexing; floating-point tolerances are not the proof of exact CI.

Use \(R\) independent Poisson batches, with \(R\) odd, and take a majority
decision for each query. If its base error is at most \(1/3\), its majority
error is at most
\[
\eta_R=\sum_{r=(R+1)/2}^{R}\binom Rr(1/3)^r(2/3)^{R-r}
      =I_{1/3}((R+1)/2,(R+1)/2).
\]
For \(Q_{\rm tests}\) possible queries, choose \(R\) so that
\(Q_{\rm tests}\eta_R\le\delta\). Then a union bound ensures, with probability
at least \(1-\delta\), that every true-edge query rejects CI and a selected
genuine separator for every missing edge accepts CI. That event implies
\(\widehat G=G\) under delete-on-any-separator recovery.

Not every nonedge query has to be either exactly CI or separated by the
chosen \(\varepsilon\). A weakly dependent nonseparator for a missing edge
may lie in the grey zone; its answer cannot prevent deletion by that edge's
genuine separator. This is a sufficient event for these simulations rather
than a general relaxation of the graph-recoverability condition. Batches must be independent across
the \(R\) repetitions; queries may reuse each batch because the union bound
does not require independence across queries.
