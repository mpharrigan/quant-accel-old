MSM Accelerator Toy-Movie Analysis
===========

Quantitative analysis of msmaccelerator on the muller potential.

A long 'gold standard' trajectory was simulated. The final value of the first implied timescale was used
as a benchmark. The error (difference between the implied timescale and the 'gold' implied timescale) over time
was fitted for the long run (without adaptive sampling) as well as for the adaptive sampling methods.

Speedup is defined as the walltime gain by using the adaptive sampling method given an error cutoff.

The following shows speedup vs length-per-trajectory. This compares short-many vs long-few. A length per trajectory
lower than 1000 gave insufficient counts for building a reliable MSM

![lpt](https://raw.github.com/mpharrigan/toy-movie-2/master/lpt.png)

The following shows speedup vs. number of parallel simulations

![ll](https://raw.github.com/mpharrigan/toy-movie-2/master/ll.png)

The following shows error over time with uncertainty. An adaptive simulation was run 100 times
to gather statistics. $error = abs(ln(IT_gold / IT_calc))$

![stat](https://raw.github.com/mpharrigan/toy-movie-2/master/stat.png)

The derivation of these plots is in the included IPython notebook, Toy-Movie-Stats-2.ipynb

