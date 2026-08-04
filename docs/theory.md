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

## Effective reference sample size

SpatialSHAP reports

\[
n_{eff,i}=\frac{1}{\sum_r w_{ir}^2}
\]

to reveal whether a nominally large background is effectively dominated by only a
few observations.

## Current estimator

Version 0.0.1 enumerates all \(2^p\) coalitions. This is a numerical oracle for
method validation, not the final scalable estimator.
