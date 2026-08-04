# Theory

## Spatially conditioned value function

For a focal observation \(i\), feature coalition \(S\), prediction function \(f\),
and location-dependent empirical reference distribution \(Q_i\), define

\[
v_i(S)=\mathbb{E}_{X_{\bar S}\sim Q_i}
\left[f(x_{i,S},X_{\bar S})\right].
\]

For background observations \(x_r\) with normalized weights \(w_{ir}\), the
implemented estimator is

\[
\widehat v_i(S)=\sum_r w_{ir}
f(x_{i,S},x_{r,\bar S}).
\]

Feature \(j\)'s exact value is

\[
\phi_{ij}=\sum_{S\subseteq N\setminus\{j\}}
\frac{|S|!(p-|S|-1)!}{p!}
\left[\widehat v_i(S\cup\{j\})-\widehat v_i(S)\right].
\]

Because a fixed \(Q_i\) defines an ordinary cooperative game for each focal
observation, the standard efficiency, symmetry, dummy, and additivity properties
apply to that local game.

## Local baseline

The empty-coalition value

\[
\phi_{0,i}=\widehat v_i(\varnothing)
\]

is location dependent whenever the reference weights vary by focal location.
Therefore

\[
f(x_i)=\phi_{0,i}+\sum_j\phi_{ij}
\]

is a locally referenced additive decomposition, not a decomposition around one
universal model expectation.

## Joint geographic reference-switch game

The joint decomposition uses \(p\) ordinary feature players and one grouped player
\(G\), called `GEO`. The GEO player is not passed to the prediction function as a
fabricated coordinate feature. Instead, it controls which empirical reference
measure supplies absent feature values.

For joint coalition \(A\subseteq N\cup\{G\}\), let \(A_X=A\cap N\). Define

\[
V_i(A)=\sum_r \omega_{ir}(A)
 f(x_{i,A_X},x_{r,N\setminus A_X}),
\]

where

\[
\omega_{ir}(A)=
\begin{cases}
\bar w_r, & G\notin A,\\
w_{ir}, & G\in A.
\end{cases}
\]

Here \(\bar w_r\) is the global empirical reference weight and \(w_{ir}\) is the
focal location's spatial reference weight. Consequently, the empty coalition has
one global baseline

\[
\phi_0=V_i(\varnothing),
\]

while the GEO player measures the contribution of changing the counterfactual
population from global to local.

## Four-component decomposition

For a coalition indicator vector \(z\), SpatialSHAP represents the joint game with

\[
V_i(A)-V_i(\varnothing)
\approx
\sum_{j=1}^{p}z_j\beta_{ij}
+z_G\beta_{iG}
+\sum_{j=1}^{p}z_jz_G\beta_{ijG}.
\]

The terms are interpreted as:

- \(\beta_{ij}\): feature \(j\)'s primary contribution under the global reference;
- \(\beta_{iG}\): intrinsic GEO main contribution from the reference switch;
- \(\beta_{ijG}\): the change in feature \(j\)'s contribution associated with that
  reference switch.

Intermediate coalitions are fitted with the finite SHAP kernel

\[
\pi(A)=
\frac{M-1}{\binom{M}{|A|}|A|(M-|A|)},
\qquad M=p+1.
\]

The full-coalition efficiency equation is imposed as a hard equality constraint:

\[
\sum_j\beta_{ij}+\beta_{iG}+\sum_j\beta_{ijG}
=V_i(N\cup\{G\})-V_i(\varnothing).
\]

Therefore the exact four-component reconstruction is

\[
f(x_i)=
\phi_0+
\sum_j\beta_{ij}+
\beta_{iG}+
\sum_j\beta_{ijG}.
\]

## Recovery of ordinary joint Shapley values

The ordinary \(p+1\)-player Shapley allocation is recovered by sharing each
GEO–feature interaction equally between its two participating players:

\[
\phi_{ij}^{joint}=\beta_{ij}+\frac{1}{2}\beta_{ijG},
\]

\[
\phi_{iG}^{joint}=\beta_{iG}+\frac{1}{2}\sum_j\beta_{ijG}.
\]

The implementation verifies this equivalence numerically for every explained
observation and reports the maximum discrepancy.

## Effective reference sample size

SpatialSHAP reports

\[
n_{eff,i}=\frac{1}{\sum_r w_{ir}^2}
\]

to reveal whether a nominally large background is effectively dominated by only a
few observations.

## Interpretation boundary

The four-component basis is a structured representation of one reference-switch
cooperative game. It is not a causal decomposition, a spatially varying
coefficient model, or a claim that coordinates physically generate the response.
When the joint game contains higher-order structure, the restricted basis absorbs
that structure into its fitted primary, GEO, and GEO–feature terms. These terms
must therefore be interpreted together with the model, reference population,
distance definition, kernel, and bandwidth.

## Current estimators

Version 0.0.1 enumerates all \(2^p\) coalitions for the conditional explainer and
all \(2^{p+1}\) coalitions for the joint geographic explainer. These exact
implementations are numerical oracles for method validation, not the final
scalable estimators.
