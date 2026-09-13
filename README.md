# BayesJL

BayesJL is a Julia package for Bayesian inference with Markov chain Monte Carlo (MCMC), transformed parameters, diagnostics, and inverse-importance evidence estimation.

## Installation

From the Julia package manager:

```julia
using Pkg
Pkg.add(url = "https://github.com/CVerhoosel/BayesJL")
```

For local development, clone the repository and activate the project:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Basic workflow

BayesJL provides distributions and transformations for defining parameter spaces, MCMC sampling utilities, and diagnostics for inspecting chains. A typical workflow is:

```julia
using BayesJL, Distributions

parameters = ParameterDistribution(θ = Normal(0, 1))
y0 = sample_starting_points(parameters, 4)
log_probability = θ -> logpdf(parameters, θ)
chain, log_probability_values = sample_distribution(
    log_probability,
    4,
    1_000,
    y0,
    100,
)

print_diagnostics(chain; param_names = keys(parameters))
```

See the scripts in [`examples/`](examples/) for complete examples.

## Testing

Run the package tests from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Documentation

Build the documentation from the repository root with:

```bash
julia --project=docs docs/make.jl
```

The generated site is written to `docs/build/`.

## License

BayesJL is licensed under the MIT License. See [`LICENSE`](LICENSE).
