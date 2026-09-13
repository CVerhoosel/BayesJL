"""
BayesJL

A Julia package for Bayesian Markov Chain Monte Carlo analysis.
"""
module BayesJL

    using Distributions, Random, OrderedCollections

    include("transforms.jl")
    include("distributions.jl")
    include("models.jl")
    include("mcmc.jl")
    include("diagnostics.jl")

    # Exports from "transforms.jl"
    export Transform, LogTransform, BoundedTransform, inverse, jacobian, logjacobian

    # Exports from "distributions.jl"
    export lognormal_from_moments, TransformedDistribution, ParameterDistribution, inverse

    # Exports from "models.jl"
    export ModelClass, CovarianceSpec, ConstantCovariance, FunctionalCovariance, ProbabilisticError, evaluate_covariance

    # Exports from "mcmc.jl"
    export logposterior, sample_starting_points, sample_distribution, flatten_chain, evidence_inverse_importance, log_evidence_inverse_importance, compute_plausibilities
    
    # Exports from "diagnostics.jl"
    export gelman_rubin, autocorrelation, autocorrelation_exp, autocorrelation_time, autocorrelation_time_exp, effective_sample_size, effective_sample_size_exp, print_diagnostics, compute_diagnostics, within_chain_stats
    export pooledvar, vario, logmeanexp, logvarexp, logpooledvarexp, logvarioexp

end # module BayesJL