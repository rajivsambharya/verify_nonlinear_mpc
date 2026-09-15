# verify_nonlinear_mpc

This repository is by
[Rajiv Sambharya](https://rajivsambharya.github.io/), [Sribalaji C. Anand](https://sites.google.com/view/sribalajianand/home), and [George Pappas](https://www.georgejpappas.org/).
It contains the Python source code to reproduce the experiments in our paper
"[Verifying performance, stability, and feasibility of inexact non-linear model predictive controllers](https://arxiv.org/pdf/2609.14920)."

## Installation
To install the verify_nonlinear_mpc package, run
```
git clone https://github.com/rajivsambharya/verify_nonlinear_mpc.git
pip install -e ".[dev]"
```

## Running experiments
Experiments can be run from using the following command:
```
python benchmarks/run_experiment.py <example> local
```

Replace the ```<example> ``` with one of the following to run an experiment.
```
nonlinear_double_integrator
stir_tank_subopt
bilinear_closed_loop
cartpole
two_tank_constrained
```

The specifics in each experiment match those in our paper. Adapt the config files (in ```benchmarks/configs```) to test other configurations.