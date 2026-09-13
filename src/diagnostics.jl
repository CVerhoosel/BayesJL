using Printf, Statistics, LogExpFunctions

"""
    _dims_tuple(chain, dims)

Convert a `dims` argument (Colon, Int, or Tuple) to a tuple of dimension indices.

# Arguments
- `chain`: Array to get dimensions from
- `dims::Union{Colon, Int, Tuple}`

# Returns
- `Tuple{Int, ...}`: Tuple of dimension indices
"""
function _dims_tuple(chain; dims::Union{Colon, Int, Tuple}=:)
    if dims === Colon()
        return (1:ndims(chain)...,)
    elseif isa(dims, Integer)
        return (dims,)
    else
        return dims
    end
end

"""
    _logN(lchain, dims)

Compute the log of the total number of elements along specified dimensions. Equivalent to
`log(prod(size(lchain, d) for d in dims))`.

# Arguments
- `lchain`: Array to get dimensions from
- `dims::Union{Colon, Int, Tuple}`: Dimension specification

# Returns
- `Float64`: log(N) where N is the product of sizes along specified dimensions
"""
function _logN(lchain; dims::Union{Colon, Int, Tuple}=:)
    return sum(log(size(lchain, d)) for d in _dims_tuple(lchain; dims=dims))
end

"""
    _logabsdiffexp(la, lb)

Compute `log(|exp(la)-exp(lb)|)`.
"""
function _logabsdiffexp(la::Float64, lb::Float64)
    if la > lb
        return la + log1mexp(lb - la)
    elseif la < lb
        return lb + log1mexp(la - lb)
    else
        return -Inf    
    end
end

"""
    logmeanexp(lchain; dims)

Compute `log(mean(exp(lchain)))` along the provided dimensions using a stable
`logsumexp` reduction.
"""
function logmeanexp(lchain; dims::Union{Colon, Int, Tuple}=:)
    lN = _logN(lchain; dims=dims)
    return logsumexp(lchain; dims=dims) .- lN
end

"""
    logvarexp(lchain; dims)

Let `var(cᵢ) = 1/(N-1) Σᵢ ( c_i - μ )^2` then `logvarexp(cᵢ) = lN + 2 lμ - log(exp(lN)-1) + log(exp(lβ)-1)`
with `β = 1/N Σᵢ ( c_i^2 / μ^2 )`.

Compute `log(var(exp(lchain)))` along the provided dimensions using a stable
log-space formulation.
"""
function logvarexp(lchain; dims::Union{Colon, Int, Tuple}=:)
    lN = _logN(lchain; dims=dims)
    lμ = logmeanexp(lchain; dims=dims)

    lβ = logmeanexp(2*(lchain .- lμ); dims=dims)
    lvar = lN .+ 2lμ .- logexpm1.(lN) .+ logexpm1.(lβ)

    return lvar
end

"""
    vario(chain; dims::Int=ndims(chain), maxlag::Int = size(chain, dims) - 1)
"""
function vario(chain; dims::Int=ndims(chain), maxlag::Int = size(chain, dims) - 1)

    N = size(chain, dims)
    V = zeros( ((d==dims ? maxlag : size(chain, d)) for d in 1:ndims(chain))... )
    for t in 1:maxlag
        chain_one_to_Nmt = chain[((d==dims ? UnitRange(1, N-t) : Colon()) for d in 1:ndims(chain))...]
        chain_t_to_N     = chain[((d==dims ? UnitRange(t+1, N) : Colon()) for d in 1:ndims(chain))...]
        chainΔ² = (chain_one_to_Nmt - chain_t_to_N).^ 2
        V[((d==dims ? UnitRange(t, t) : Colon()) for d in 1:ndims(chain))...] = mean(chainΔ²; dims=dims)
    end

    return V
end

function logvarioexp(lchain; dims::Int=ndims(lchain), maxlag::Int = size(lchain, dims) - 1)

    N  = size(lchain, dims)
    lV = zeros( ((d==dims ? maxlag : size(lchain, d)) for d in 1:ndims(lchain))... )
    for t in 1:maxlag
        lchain_one_to_Nmt = lchain[((d==dims ? UnitRange(1, N-t) : Colon()) for d in 1:ndims(lchain))...]
        lchain_t_to_N     = lchain[((d==dims ? UnitRange(t+1, N) : Colon()) for d in 1:ndims(lchain))...]
        lchainΔ² = 2*_logabsdiffexp.(lchain_one_to_Nmt, lchain_t_to_N)
        lV[((d==dims ? UnitRange(t, t) : Colon()) for d in 1:ndims(lchain))...] = logmeanexp(lchainΔ²; dims=dims)
    end

    return lV
end

"""
    pooledvar(chain)

Compute the pooled variance estimator var⁺ for each parameter in an MCMC chain.

# Arguments
- `chain::Array{Float64,3}`

# Returns
- `var⁺::Vector{Float64}`: Variance estimates for each parameter
"""
function pooledvar(chain::Array{Float64,3})
    # Samples per walker
    N = size(chain, 3)

    # Within-chain variance
    W = mean(var(chain; dims=3); dims=2)

    # Between-chain variance
    B = var(mean(chain; dims=3); dims=2) * N

    # Pooled variance estimator
    var⁺ = ( (N - 1) / N ) * W .+ (1 / N) * B

    return var⁺
end

function pooledvar(chain::Array{Float64,2})
    return pooledvar(reshape(chain, (1, size(chain)...)))[1]
end

function logpooledvarexp(lchain::Array{Float64,3})
    # Samples per walker
    lN = _logN(lchain; dims=3)

    # Within-chain variance
    lW = logmeanexp(logvarexp(lchain; dims=3); dims=2)

    # Between-chain variance
    lB = logvarexp(logmeanexp(lchain; dims=3); dims=2) .+ lN

    # Pooled variance estimator
    lβ = logexpm1(lN) .+ lW .- lB
    lvar⁺ = lB .- lN .+ log1pexp.(lβ)
    
    return lvar⁺
end

function logpooledvarexp(lchain::Array{Float64,2})
    return logpooledvarexp(reshape(lchain, (1, size(lchain)...)))[1]
end

"""
    gelman_rubin(chain)

Compute the Gelman-Rubin convergence diagnostic R̂ for each parameter in an MCMC chain.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, N)

# Returns
- `R̂::Vector{Float64}`: Gelman-Rubin statistic for each parameter (length p)
"""
function gelman_rubin(chain::Array{Float64,3})
    p, M, N = size(chain)
    
    @assert M ≥ 2 "Gelman-Rubin diagnostic requires at least 2 chains"
    
    # Within-chain variance (averaged across chains)
    W = mean(var(chain; dims=3); dims=2)
    
    # Pooled variance estimator
    var⁺ = pooledvar(chain)
    
    # Gelman-Rubin statistic
    R̂ = sqrt.(var⁺ ./ W)
    
    return R̂
end

"""
    autocorrelation(chain)

Compute the autocorrelation estimator ρ̂ for each parameter in an MCMC chain.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, N)

# Returns
- `ρ::Matrix{Float64}`: Autocorrelation estimates with shape (p, N-1)
"""
function autocorrelation(chain::Array{Float64,3}; maxlag::Int = size(chain, 3) - 1)
    
    # Pooled variance and vario
    var⁺ = pooledvar(chain)
    Vₜ = mean(vario(chain; dims=3, maxlag=maxlag); dims=2)

    # Compute autocorrelation ρᵢ,ₜ for each parameter and lag
    p = size(chain, 1)
    ρ = Matrix{Float64}(undef, p, maxlag)
    for i in 1:p
        ρ[i,:] = 1.0 .- (0.5/var⁺[i]) * Vₜ[i, 1, :]
    end
    return ρ
end

function autocorrelation(chain::Array{Float64,2}; maxlag::Int = size(chain, 2) - 1)
    return autocorrelation(reshape(chain, (1, size(chain)...)); maxlag=maxlag)[1]
end

function autocorrelation_exp(lchain::Array{Float64,3}; maxlag::Int = size(lchain, 3) - 1)
    
    # Pooled variance and vario
    lvar⁺ = logpooledvarexp(lchain)
    lVₜ = logmeanexp(logvarioexp(lchain; dims=3, maxlag=maxlag); dims=2)

    # Compute autocorrelation ρᵢ,ₜ for each parameter and lag
    p = size(lchain, 1)
    ρ = Matrix{Float64}(undef, p, maxlag)
    for i in 1:p
        ρ[i,:] = 1.0 .- exp.(log(0.5) .- lvar⁺[i] .+  lVₜ[i, 1, :])
    end
    return ρ
end

function autocorrelation_exp(lchain::Array{Float64,2}; maxlag::Int = size(lchain, 2) - 1)
    return autocorrelation_exp(reshape(lchain, (1, size(lchain)...)); maxlag=maxlag)[1]
end

"""
    autocorrelation_time(chain, maxlag::Int = size(chain, 3) - 1)

Compute the integrated autocorrelation time τ for each parameter in an MCMC chain.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, N)
- `maxlag::Int`: Maximum lag to consider (default: N-1)

# Returns
- `τ::Vector{Float64}`: Autocorrelation time for each parameter (length p)
"""
function autocorrelation_time(chain::Array{Float64,3}; maxlag::Int = size(chain, 3) - 1)
    p = size(chain, 1)
    
    # Compute autocorrelation
    ρ = autocorrelation(chain; maxlag=maxlag)

    # Compute integrated autocorrelation time for each parameter
    τ = ones(p)
    for i in 1:p
        for t in 1:maxlag
            if ρ[i, t] < 0
                break # Truncate when autocorrelation becomes negative
            end
            τ[i] += 2.0 * ρ[i, t]
        end
    end
    
    return τ
end

function autocorrelation_time(chain::Array{Float64,2}; maxlag::Int = size(chain, 2) - 1)
    return autocorrelation_time(reshape(chain, (1, size(chain)...)); maxlag=maxlag)[1]
end

function autocorrelation_time_exp(lchain::Array{Float64,3}; maxlag::Int = size(lchain, 3) - 1)
    p = size(lchain, 1)
    
    # Compute autocorrelation
    ρ = autocorrelation_exp(lchain; maxlag=maxlag)
    
    # Compute integrated autocorrelation time for each parameter
    τ = ones(p)
    for i in 1:p
        for t in 1:maxlag
            if ρ[i, t] < 0
                break # Truncate when autocorrelation becomes negative
            end
            τ[i] += 2.0 * ρ[i, t]
        end
    end
    
    return τ
end

function autocorrelation_time_exp(lchain::Array{Float64,2}; maxlag::Int = size(lchain, 2) - 1)
    return autocorrelation_time_exp(reshape(lchain, (1, size(lchain)...)); maxlag=maxlag)[1]
end

"""
    effective_sample_size(chain, maxlag::Int = size(chain, 3) - 1)

Compute the effective sample size (ESS) for each parameter in an MCMC chain.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, Nₘ)
- `maxlag::Int`: Maximum lag to consider (default: Nₘ-1)

# Returns
- `N̄::Vector{Float64}`: Effective sample size for each parameter (length p)
"""
function effective_sample_size(chain::Array{Float64,3}; maxlag::Int = size(chain, 3) - 1)
    _, M, Nₘ = size(chain)
    
    # Compute autocorrelation time
    τ = autocorrelation_time(chain; maxlag=maxlag)
    
    # Effective sample size
    N̄ = (M * Nₘ) ./ τ
    
    return N̄
end

function effective_sample_size(chain::Array{Float64,2}; maxlag::Int = size(chain, 2) - 1)
    return effective_sample_size(reshape(chain, (1, size(chain)...)); maxlag=maxlag)[1]
end

function effective_sample_size_exp(lchain::Array{Float64,3}; maxlag::Int = size(lchain, 3) - 1)
    _, M, Nₘ = size(lchain)
    
    # Compute autocorrelation time
    τ = autocorrelation_time_exp(lchain; maxlag=maxlag)
    
    # Effective sample size
    N̄ = (M * Nₘ) ./ τ
    
    return N̄
end

function effective_sample_size_exp(lchain::Array{Float64,2}; maxlag::Int = size(lchain, 2) - 1)
    return effective_sample_size_exp(reshape(lchain, (1, size(lchain)...)); maxlag=maxlag)[1]
end

"""
    compute_diagnostics(chain; maxlag::Int = 200, param_names=nothing)

Compute MCMC diagnostics for each parameter in a chain.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, N)
- `maxlag::Int`: Maximum lag for autocorrelation calculations (default: 200)
- `param_names`: Optional list (Vector or Tuple) of parameter names as Symbols or Strings (default: θ₁, θ₂, ...)

# Returns
- Named tuple of named tuples, where each parameter has fields: μ, σ, τ, N̄, R̂
```
"""
function compute_diagnostics(chain::Array{Float64,3}; maxlag::Int = size(chain, 3)÷10, param_names=nothing)
    p = size(chain, 1)
    
    # Compute posterior statistics (flatten chain for mean/std)
    means = mean(chain, dims=(2,3))
    stds = std(chain, dims=(2,3))

    # Compute diagnostics
    R̂ = gelman_rubin(chain)
    τ = autocorrelation_time(chain; maxlag=maxlag)
    N̄ = effective_sample_size(chain; maxlag=maxlag)
    
    # Generate/normalize parameter names
    if param_names === nothing
        param_names = ["θ_$i" for i in 1:p]
    else
        # Accept Symbols, Strings, Tuples, Vectors; normalize to Vector{String}
        param_names = collect(string.(param_names))
    end
    
    # Construct named tuple of diagnostics for each parameter
    d = NamedTuple{Tuple(Symbol.(param_names))}(
        tuple((
            (μ = means[i], σ = stds[i], τ = τ[i], N̄ = N̄[i], R̂ = R̂[i])
            for i in 1:p
        )...)
    )
    
    return d
end

"""
    print_diagnostics(chain; maxlag::Int = 200, param_names=nothing)

Print a formatted table of MCMC diagnostics for each parameter.

# Arguments
- `chain::Array{Float64,3}`: MCMC chain with shape (p, M, N)
- `maxlag::Int`: Maximum lag for autocorrelation calculations (default: 200)
- `param_names`: Optional list (Vector or Tuple) of parameter names as Symbols or Strings (default: θ₁, θ₂, ...)

# Returns
- Named tuple of named tuples with diagnostics for each parameter
"""
function print_diagnostics(chain::Array{Float64,3}; maxlag::Int = 200, param_names=nothing)
    p, M, Nₘ = size(chain)
    
    # Compute diagnostics
    d = compute_diagnostics(chain; maxlag=maxlag, param_names=param_names)
    
    # Print header
    println("\n" * "="^80)
    println("MCMC Diagnostics Summary")
    println("="^80)
    println("Chains: $M  |  Samples per chain: $Nₘ  |  Total samples: $(M*Nₘ)")
    println("="^80)
    
    # Column headers
    println(@sprintf("%-12s %10s %10s %12s %12s %12s", 
            "Parameter", "Mean", "Std", "τ", "N̄", "R̂"))
    println("-"^80)
    
    # Print each parameter's diagnostics
    for (param_name, diag) in pairs(d)
        println(@sprintf("%-12s %12.4e %12.4e %10.2f %12.1f %10.4f",
                param_name, diag.μ, diag.σ, diag.τ, diag.N̄, diag.R̂))
    end
    
    # Only print standard deviation row if there are multiple parameters
    if p > 1
        # Separator line before summary statistics
        println("-"^80)
        
        # Compute mean and std across parameters
        mean_τ = mean(getfield.(values(d), :τ))
        mean_N̄ = mean(getfield.(values(d), :N̄))
        mean_R̂ = mean(getfield.(values(d), :R̂))

        std_τ = std(getfield.(values(d), :τ))
        std_N̄ = std(getfield.(values(d), :N̄))
        std_R̂ = std(getfield.(values(d), :R̂))
        
        # Print summary rows
        println(@sprintf("%-12s %12s %12s %10.2f %12.1f %10.4f",
            "Mean", "", "", mean_τ, mean_N̄, mean_R̂))
        println(@sprintf("%-12s %12s %12s %10.2f %12.1f %10.4f",
            "Std", "", "", std_τ, std_N̄, std_R̂))
    end
    
    println("="^80)
    
    # Print warnings for convergence issues
    convergence_issues = false
    for (param_name, param_d) in pairs(d)
        if param_d.R̂ > 1.1
            if !convergence_issues
                println("\n⚠ Convergence Warnings:")
                convergence_issues = true
            end
            println(@sprintf("  • %s: R̂ = %.4f (> 1.1) - chain may not have converged", 
                    param_name, param_d.R̂))
        end
    end
    
    if !convergence_issues
        println("\n✓ All parameters show good convergence (R̂ ≤ 1.1)")
    end
    
    println()
    
    return d
end

"""
        within_chain_stats(chain)

Compute the mean vector and covariance matrix by pooling samples within each chain
and then averaging across chains.

# Arguments
- `chain::Array{Float64,3}`: MCMC samples with shape `(p, M, Nᵐ)` where
    - `p` = number of parameters
    - `M` = number of chains/walkers
    - `Nᵐ` = samples per chain

# Returns
- `(μ, Σ)`: Tuple containing
    - `μ::Vector{Float64}` length `p`, the mean of each parameter averaged over chains
    - `Σ::Matrix{Float64}` `p×p`, the covariance averaged over chains
"""
function within_chain_stats(chain::Array{Float64,3})
    p, M, _ = size(chain)

    # Within chain statistics
    μ = zeros(p)
    Σ = zeros(p, p)
    for m in 1:M
        μ += mean(chain[:,m,:]; dims=2)[:,1]
        Σ += cov(chain[:,m,:]; dims=2)
    end
    μ ./= M
    Σ ./= M

    return μ, Σ
end