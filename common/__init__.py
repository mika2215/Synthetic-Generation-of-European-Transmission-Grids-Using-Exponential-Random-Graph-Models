"""Shared code for all analysis steps.

Modules
-------
paths       : locations of input data, stored results and figures
grid        : construction of the country graphs from the PyPSA-Eur tables
statistics  : observed statistics and change statistics of the ERGM terms
ee          : equilibrium-expectation (EE) estimation
sampler     : Metropolis-Hastings sampler restricted to connected graphs
thinning    : spacing between retained samples, s(n)
robustness  : edge-removal curves S(f) and the robustness index R
"""
