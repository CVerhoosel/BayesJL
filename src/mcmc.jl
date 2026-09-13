import AffineInvariantMCMC
using Clustering
using LinearAlgebra
using Statistics
using SpecialFunctions

"""
    logposterior(parameters, model, prior, scenarios, ε)

Compute the log posterior probability for a given set of parameters.

The log posterior is the sum of the log prior and log likelihood:
    log p(θ|y) = log p(θ) + log p(y|θ)

# Arguments
- `parameters::Vector`: Parameter values in the transformed (unbounded) space
- `model::ModelClass`: Wrapped model function that computes predictions
- `prior::ParameterDistribution`: Prior distribution over parameters
- `scenarios::Dict`: Dictionary of experimental scenarios and observations
- `ε::ProbabilisticError`: Error model combining measurement noise and model bias

# Returns
- `Float64`: Log posterior probability

# Notes
Errors during evaluation are logged and re-thrown for debugging.
"""
function logposterior(parameters, model, prior, scenarios, ε)
    try
        lp  = logpdf(prior, parameters)
        lp += loglikelihood(parameters, model, prior, scenarios, ε)
        return lp
    catch e
        @error "Error in logposterior" exception=(e, catch_backtrace())
        rethrow()
    end
end

"""
    loglikelihood(parameters, model, prior, scenarios, ε)

Compute the log likelihood for a given set of parameters across all scenarios.

For each scenario, this function:
1. Transforms parameters from unbounded to bounded space
2. Computes model predictions for the scenario
3. Evaluates the probability of observations given predictions and error model
4. Sums log-probabilities across all observations and scenarios

# Arguments
- `parameters::Vector`: Parameter values in the transformed (unbounded) space
- `model::ModelClass`: Wrapped model function that computes predictions
- `prior::ParameterDistribution`: Prior distribution (used for parameter transformations)
- `scenarios::Dict`: Dictionary mapping scenario names to (γ̇, H, y) tuples
- `ε::ProbabilisticError`: Error model with measurement noise and model bias covariances

# Returns
- `Float64`: Log-likelihood summed over all scenarios and observations
"""
function loglikelihood(parameters, model, prior, scenarios, ε)
    transformed_parameters = NamedTuple{keys(prior)}((isa(d, TransformedDistribution) ? inverse(d.transform, p) : p) for (p,d) in zip(parameters, values(prior)))
    try        
        llhood = 0.
        for scenario_data in values(scenarios)
            # Compute model predictions for this scenario
            dᵢ = model(;scenario_data..., transformed_parameters...)
            
            for yᵢ in scenario_data.y
                llhood += logpdf(ε, yᵢ - dᵢ; scenario_data..., transformed_parameters...)
            end
        end

        return llhood
    catch e
        println("Likelihood evalutation failed for parameters:")
    
        for (name, value) in pairs(transformed_parameters)
            println("  $name = $value")
        end
        for (name, value) in pairs(scenarios)
            println("Scenario $name: $value")
        end

        @error "Error in loglikelihood" exception=(e, catch_backtrace())
        rethrow()
    end
end

"""
    sample_starting_points(prior, Nwalkers; Ncandidates=1000, method=:random)

Generate informed starting points for MCMC walkers using clustering.

# Arguments
- `prior::ParameterDistribution`: Prior distribution to sample from
- `Nwalkers::Int`: Number of MCMC walkers (number of clusters)

# Keyword Arguments
- `Ncandidates::Int=1000`: Number of candidate samples to draw before clustering
- `method::Symbol=:random`: Clustering method `:kmeans` or `:random`

# Returns
- `Matrix`: Starting points with shape (p × Nwalkers)

# Examples
```julia
# Default: 1000 candidates, k-means clustering
y0 = sample_starting_points(prior, 10; method=:kmeans)

# More candidates for better coverage
y0 = sample_starting_points(prior, 10;  method=:kmeans, Ncandidates=5000)

# Fallback to random sampling
y0 = sample_starting_points(prior, 10; method=:random)
```
"""
function sample_starting_points(prior, Nwalkers; Ncandidates=100*Nwalkers, method=:random)

    if method == :random
        return mean(rand(prior, Nwalkers, Ncandidates÷Nwalkers), dims=3)[:,:,1]
    end
    
    # Sample many candidates from the prior
    candidates = rand(prior, Ncandidates)
    
    # Perform k-means clustering
    result = kmeans(candidates, Nwalkers; maxiter=100, display=:none)
    
    # Return cluster centers as starting points
    return result.centers
end

"""
    sample_distribution(logprobability, Nwalkers, Nsamples_per_walker, y0, Nburnin)

Run affine-invariant ensemble MCMC sampling with burn-in.

# Arguments
- `logprobability::Function`: Function that takes parameters and returns log probability
- `Nwalkers::Int`: Number of parallel walkers in the ensemble
- `Nsamples_per_walker::Int`: Number of samples to collect per walker (after burn-in)
- `y0::Matrix`: Starting positions with shape (p × Nwalkers)
- `Nburnin::Int`: Number of burn-in steps to discard

# Returns
- `chain`: MCMC chain with shape (p, Nwalkers, Nsamples_per_walker)
- `lpvals`: Log probability values with shape (Nwalkers, Nsamples_per_walker)
"""
function sample_distribution(logprobability, Nwalkers, Nsamples_per_walker, y0, Nburnin)
    chain, lpvals = AffineInvariantMCMC.sample(logprobability, Nwalkers, y0, Nburnin, 1)
    chain, lpvals = AffineInvariantMCMC.sample(logprobability, Nwalkers, chain[:, :, end], Nsamples_per_walker, 1)
end

"""
    flatten_chain(chain, lpvals)

Flatten MCMC chain from 3D (parameters × walkers × samples) to 2D (parameters × total_samples).

# Arguments
- `chain::Array{3}`: MCMC chain with shape (p, M, Nᵐ)
- `lpvals::Matrix`: Log probability values with shape (M, Nᵐ)

# Returns
- `flatchain::Matrix`: Flattened chain with shape (p, M × Nᵐ)
- `flatlpvals::Vector`: Flattened log probabilities with length M × Nᵐ
"""
function flatten_chain(chain, lpvals)
    flatchain, flatlpvals = AffineInvariantMCMC.flattenmcmcarray(chain, lpvals)
    return flatchain, flatlpvals
end

"""
    evidence_inverse_importance(chain, lpvals, prior; R=1.0)

Estimate the inverse model evidence z^{-1} via importance sampling using a
uniform density φ over a mean-centered ellipsoid of radius R in the posterior
transformed parameter space.

Inputs:
 - chain: 3D array (p × M × Nₘ)
 - lpvals: logposterior values (M × Nₘ)
 - R: radius of ellipsoid (default 1.0)

# Returns
- `ẑ⁻¹::Normal`: Normal distribution for the inverse evidence estimator
- `f::Float64`: Fraction of samples inside the ellipsoid
- `N̄::Int`: Effective sample size
"""
function evidence_inverse_importance(chain, lp; R=2.0, maxlag=size(chain, 3) - 1)
    lẑ⁻¹, f, N̄ = _log_evidence_inverse_importance(chain, lp; R=R, maxlag=maxlag)
    ẑ⁻¹ = LogNormal(lẑ⁻¹...)
    return ẑ⁻¹, f, N̄
end

function log_evidence_inverse_importance(chain, lp; R=2.0, maxlag=size(chain, 3) - 1)
    lẑ⁻¹, f, N̄ = _log_evidence_inverse_importance(chain, lp; R=R, maxlag=maxlag)
    return lẑ⁻¹, f, N̄
end

function _log_evidence_inverse_importance(chain, lp; R=R, maxlag=maxlag)
    
    lφ, f = _ellipsoid_logpdf(chain, R)

    if f ≈ 0.
        return nothing, f, 0
    end

    # Importance sampled inverse logprobability
    lq = lφ .- lp

    lμq   = logmeanexp(lq)
    lvarq = logvarexp(lq)

    N̄ = effective_sample_size_exp(lq; maxlag=maxlag)
    lN̄ = log(N̄)

    lμẑ⁻¹ = lμq
    lσẑ⁻¹ = 0.5*(lvarq - lN̄)

    varlẑ⁻¹ = log1pexp(2*lσẑ⁻¹-2*lμẑ⁻¹)
    μlẑ⁻¹   = lμẑ⁻¹ - 0.5 * varlẑ⁻¹
    σlẑ⁻¹   = sqrt(varlẑ⁻¹)  

    lẑ⁻¹ = (μlẑ⁻¹, σlẑ⁻¹)

    return lẑ⁻¹, f, N̄
end

"""
    _ellipsoid_logpdf(flatchain, R)

Compute log probabilities for a uniform indicator distribution over a mean-centered
ellipsoid in parameter space.

# Arguments
 - chain: 3D array (p × M × Nₘ) in transformed space
 - `R::Float64`: Radius of the ellipsoid in Mahalanobis distance units

# Returns
- `lφ::Matrix{Float64}`: Log probabilities for each sample (-log(Vᵩ) if inside, -Inf if outside)
- `Vᵩ::Float64`: Volume of the ellipsoid
- `f::Float64`: Fraction of samples inside the ellipsoid
"""
function _ellipsoid_logpdf(chain, R=2.0)
    p, M, Nₘ = size(chain)

    # Within chain statistics
    μ, Σ = within_chain_stats(chain)
    
    # Ellipsoid volume V
    Vᵩ = (π^(p/2) / gamma(p/2 + 1)) * R^p * sqrt(abs(det(Σ)))
    
    # Precomputations
    Σinv = inv(Σ)
    lVᵩ = log(Vᵩ)
    
    lφ  = Matrix{Float64}(undef, M, Nₘ)
    Nin = 0
    @inbounds for m in 1:M
        for n in 1:Nₘ
            Δ = chain[:, m, n] .- μ
            q = dot(Δ, Σinv * Δ)
            if q < R^2
                lφ[m, n] = -lVᵩ
                Nin += 1
            else
                lφ[m, n] = -Inf
            end
        end
    end

    return lφ, Nin/(M*Nₘ)
end

"""
    compute_plausibilities(allz⁻¹::Tuple, Nz=10_000)

Compute model plausibilities from inverse evidence estimates.

# Arguments
- `allz⁻¹::Tuple`: Tuple of inverse evidence samples `z⁻¹` for each model. Each element
  should be a vector of samples from the inverse evidence distribution.
- `Nz::Int=10_000`: Number of Monte Carlo samples to use for computing plausibilities.

# Returns
- `ρ::Tuple`: Tuple of log-normal distributions representing the plausibility of each model.
  Each element is a distribution fitted to the Monte Carlo samples of that model's plausibility.
```
"""
function compute_plausibilities(allz⁻¹::Tuple, Nz=10_000)

    m  = length(allz⁻¹)
    lz = Matrix{Float64}(undef, Nz, m)
    for j in 1:m
        lz⁻¹ = Normal(meanlogx(allz⁻¹[j]), stdlogx(allz⁻¹[j]))
        lz[:, j] = -rand(lz⁻¹, Nz)
    end
    lρ = lz .- logsumexp(lz; dims=2) 

    return ((LogNormal(mean(lρ[:,j]), std(lρ[:,j])) for j in 1:m)...,)

end