"""
    distributions.jl

Extends Distributions.jl with transformed distributions, alternative
parameterizations, and parameter distributions.
"""

# ============================================================================
# LogNormal Alternative Constructors
# ============================================================================

"""
    lognormal_from_moments(mean::Real, std::Real)

Create a LogNormal distribution specified by the mean and standard deviation of the
lognormal random variable itself (not the parameters μ, σ of the underlying normal
distribution in log-space).

# Arguments
- `mean::Real`: Mean of the lognormal random variable (must be positive)
- `std::Real`: Standard deviation of the lognormal random variable (must be positive)

# Examples
```jldoctest
julia> using BayesJL, Distributions

julia> d = lognormal_from_moments(7.0, 3.0);

julia> round(mean(d), digits=10)
7.0

julia> round(std(d), digits=10)
3.0

julia> round(meanlogx(d), digits=10)
1.8615987928

julia> round(stdlogx(d), digits=10)
0.4106369594
```
"""
function lognormal_from_moments(mean::Real, std::Real)
    @assert mean > 0 "Mean must be positive for LogNormal distribution"
    @assert std > 0 "Standard deviation must be positive"

    σ_sq = log1p((std / mean)^2)
    σ = sqrt(σ_sq)
    μ = log(mean) - σ_sq / 2

    return LogNormal(μ, σ)
end

# ============================================================================
# TransformedDistribution
# ============================================================================

"""
    TransformedDistribution{D<:Distribution, T<:Transform}

A distribution in transformed space, combining a base distribution with a transformation.

# Fields
- `original::D`: The base distribution in the original space
- `transform::T`: The transformation to apply

# Constructors
```julia
TransformedDistribution(original, transform) # Explicit distribution and transform
TransformedDistribution(::Uniform)           # Automatic using BoundedTransform
TransformedDistribution(::LogNormal)         # Automatic using LogTransform
```

# Supported Operations
- `rand(td)`, `rand(td, n)`: Sample from the transformed distribution
- `pdf(td, y)`: Compute PDF in transformed space (with Jacobian correction)
- `mean(td)`: Compute mean in transformed space (for supported combinations)
- `var(td)`: Compute variance in transformed space (for supported combinations)
- `std(td)`: Compute standard deviation in transformed space (for supported combinations)

# Examples
```jldoctest
julia> using BayesJL, Distributions

julia> d = lognormal_from_moments(7.0, 3.0);

julia> td = TransformedDistribution(d);

julia> round(mean(td), digits=10)
1.8615987928

julia> round(std(td), digits=10)
0.4106369594
```
"""
struct TransformedDistribution{D<:Distribution, T<:Transform} <: Distribution{Univariate, Continuous}
    original::D
    transform::T
end

# Make broadcastable
Base.Broadcast.broadcastable(td::TransformedDistribution) = Ref(td)

# Convenience constructors with automatic transform selection
TransformedDistribution(o::LogNormal) = TransformedDistribution(o, LogTransform())
TransformedDistribution(o::Exponential) = TransformedDistribution(o, LogTransform())
TransformedDistribution(o::LogitNormal) = TransformedDistribution(o, BoundedTransform())
TransformedDistribution(o::Uniform) = TransformedDistribution(o, BoundedTransform(o))

# ============================================================================
# Statistical Functions
# ============================================================================

import Distributions: mean, var, std

"""
    mean(td::TransformedDistribution)

Compute the mean of a transformed distribution.

# Specialized Methods
- `LogNormal` + `LogTransform`: Uses meanlogx of the LogNormal distribution.

For unsupported combinations, raises an error.
"""
mean(td::TransformedDistribution{<:LogNormal,<:LogTransform}) = meanlogx(td.original)

mean(td::TransformedDistribution{<:Exponential,<:LogTransform}) =
    log(mean(td.original)) - Base.MathConstants.eulergamma

mean(td::TransformedDistribution) =
    error("mean not implemented for $(typeof(td)); provide a specialized method for this distribution/transform combination")

"""
    var(td::TransformedDistribution)

Compute the variance of a transformed distribution.

# Specialized Methods
- `LogNormal` + `LogTransform`: Uses varlogx of the LogNormal distribution.

For unsupported combinations, raises an error.
"""
var(td::TransformedDistribution{<:LogNormal,<:LogTransform}) = varlogx(td.original)

var(td::TransformedDistribution{<:Exponential,<:LogTransform}) = (π^2)/6

var(td::TransformedDistribution) =
    error("var not implemented for $(typeof(td)); provide a specialized method for this distribution/transform combination")

"""
    std(td::TransformedDistribution)

Compute the standard deviation of a transformed distribution.

# Specialized Methods
- `LogNormal` + `LogTransform`: Uses stdlogx of the LogNormal distribution.

For unsupported combinations, raises an error.
"""
std(td::TransformedDistribution{<:LogNormal,<:LogTransform}) = stdlogx(td.original)

std(td::TransformedDistribution{<:Exponential,<:LogTransform}) = π / sqrt(6)

std(td::TransformedDistribution) =
    error("std not implemented for $(typeof(td)); provide a specialized method for this distribution/transform combination")

# ============================================================================
# Probability Density Function
# ============================================================================

"""
    pdf(td::TransformedDistribution, y)

Compute the probability density at `y` in the transformed space.

Applies the change-of-variables formula with Jacobian correction:
    p_Y(y) = p_X(x) × dx/dy
where x = inverse(td.transform, y).
"""
function pdf_transformed(td::TransformedDistribution, y::Real)
    x = inverse(td.transform, y)
    return pdf(td.original, x) / jacobian(td.transform, x)
end

"""
    logpdf(td::TransformedDistribution, y)

Compute the log probability density at `y` in the transformed space.

Applies the change-of-variables formula with Jacobian correction:
    log p_Y(y) = log p_X(x) + log dx/dy
where x = td.transform.inverse(y).
"""
function logpdf_transformed(td::TransformedDistribution, y::Real)
    x = inverse(td.transform, y)
    return logpdf(td.original, x) - logjacobian(td.transform, x)
end
# ============================================================================
# Probability density function
# ============================================================================

import Distributions: pdf, logpdf

pdf(td::TransformedDistribution, y::Real) = pdf_transformed(td, y)
logpdf(td::TransformedDistribution, y::Real) = logpdf_transformed(td, y)

# ============================================================================
# Sampling Methods
# ============================================================================

import Random

# Tell Random.jl to use SamplerTrivial for TransformedDistribution
Random.Sampler(::Type{<:Random.AbstractRNG}, td::TransformedDistribution, ::Random.Repetition) =
    Random.SamplerTrivial(td)

"""
    rand([rng], td::TransformedDistribution)
    rand([rng], td::TransformedDistribution, n::Int)
    rand([rng], td::TransformedDistribution, dims::Int...)

Sample from the transformed distribution in 0, 1, or multiple dimensions.

Behaviors:
- `rand(td)`: returns a single transformed sample (scalar)
- `rand(td, n)`: returns a length-`n` Vector of transformed samples
- `rand(td, n1, n2, ...)`: returns an Array with the requested shape,
   applying the transform elementwise to samples from the base distribution.

Examples
```jldoctest
julia> using BayesJL, Random

julia> td = TransformedDistribution(lognormal_from_moments(7.0, 3.0));

julia> Random.seed!(42);

julia> round(rand(td), digits=10)
2.18532674

julia> round.(rand(td,3), digits=10)
3-element Vector{Float64}:
 1.5002963343
 1.5027868941
 1.5604972038

julia> round.(rand(td,3,2), digits=10)
3×2 Matrix{Float64}:
 1.57325  2.12097
 1.83179  2.45238
 1.48672  1.71616

julia> rng = MersenneTwister(42);

julia> round(rand(rng, td), digits=10)
2.3585853224

julia> round.(rand(rng, td, 3), digits=10)
3-element Vector{Float64}:
 1.8291590461
 2.0272956546
 1.980663438

julia> round.(rand(rng, td, 3, 2), digits=10)
3×2 Matrix{Float64}:
 1.83407  1.83948
 2.11372  1.62738
 2.27919  1.50938
```
"""
Random.rand(rng::Random.AbstractRNG, td::TransformedDistribution) =
    td.transform(rand(rng, td.original))

function Random.rand(rng::Random.AbstractRNG, td::TransformedDistribution, dims::Int...)
    base_samples = rand(rng, td.original, dims...)
    return td.transform.(base_samples)
end

# Also define versions without explicit RNG (uses default RNG)
Base.rand(td::TransformedDistribution) = rand(Random.default_rng(), td)
Base.rand(td::TransformedDistribution, dims::Int...) = rand(Random.default_rng(), td, dims...)

# ============================================================================
# ParameterDistribution
# ============================================================================

import Distributions: product_distribution, ProductNamedTupleDistribution

"""
    ParameterDistribution

Lightweight wrapper around `ProductNamedTupleDistribution`.

Constructors:
    ParameterDistribution(; kwargs...)
    ParameterDistribution(nt::NamedTuple{<:Any, <:Tuple{Vararg{Distribution}}})
    ParameterDistribution(od::OrderedDict{Symbol,<:Distribution})

# Examples
```jldoctest
julia> using BayesJL, Distributions, Random, OrderedCollections

julia> pd = ParameterDistribution(a = Normal(7.0, 3.0), b = Normal(5.0, 1.0));

julia> length(pd)
2

julia> keys(pd)
(:a, :b)

julia> values(pd)
(a = Normal{Float64}(μ=7.0, σ=3.0), b = Normal{Float64}(μ=5.0, σ=1.0))

julia> pd[:a]
Normal{Float64}(μ=7.0, σ=3.0)

julia> round(pdf(pd, (a=7.0, b=5.0)), digits=10)
0.0530516477

julia> round.(pdf.(pd, [[7.0,5.0], [6.0,4.0]]), digits=10)
2-element Vector{Float64}:
 0.0530516477
 0.0304385643

julia> round(logpdf(pd, (a=7.0, b=5.0)), digits=10)
-2.9364893551

julia> round.(logpdf.(pd, [[7.0,5.0], [6.0,4.0]]), digits=10)
2-element Vector{Float64}:
 -2.9364893551
 -3.4920449106

julia> Random.seed!(42);

julia> round.(rand(pd), digits=10)
2-element Vector{Float64}:
 9.3650668048
 4.120141404

julia> round.(rand(pd, 2), digits=10)
2×2 Matrix{Float64}:
 4.37862  4.89341
 4.26675  4.92742

julia> pd = ParameterDistribution(OrderedDict(:a=>Normal(7.0, 3.0), :b=> Normal(5.0, 1.0)));

julia> keys(pd)
(:a, :b)

julia> pd  = ParameterDistribution((a=Normal(7.0, 3.0), b= Normal(5.0, 1.0)));

julia> keys(pd)
(:a, :b)
```
"""
struct ParameterDistribution{P<:ProductNamedTupleDistribution}
    inner::P
end

# Constructors
function ParameterDistribution(; kwargs...)
    nt = NamedTuple(kwargs)
    @assert all(isa.(values(nt), Distribution)) "All parameters must be distributions"
    return ParameterDistribution(product_distribution(nt))
end

ParameterDistribution(nt::NamedTuple{<:Any, <:Tuple{Vararg{Distribution}}}) = ParameterDistribution(product_distribution(nt))
ParameterDistribution(od::OrderedDict{Symbol,<:Distribution}) = ParameterDistribution(product_distribution(NamedTuple(od)))

# Core accessors
Base.keys(pd::ParameterDistribution) = begin
    # Type of inner is ProductNamedTupleDistribution{names_tuple, ...}
    T = typeof(pd.inner)
    names_tuple = T.parameters[1]  # e.g. (:a, :b, :c)
    return names_tuple::Tuple{Vararg{Symbol}}
end
Base.values(pd::ParameterDistribution) = pd.inner.dists
Base.getindex(pd::ParameterDistribution, name::Symbol) = pd.inner.dists[name]
Base.length(pd::ParameterDistribution) = length(keys(pd))

# pdf / logpdf forwarding (NamedTuple)
pdf(pd::ParameterDistribution, x::NamedTuple) = pdf(pd.inner, x)
logpdf(pd::ParameterDistribution, x::NamedTuple) = logpdf(pd.inner, x)

# Vector convenience
function pdf(pd::ParameterDistribution, x::AbstractVector{<:Real})
    names = keys(pd)
    @assert length(x) == length(names) "Vector length $(length(x)) must match number of parameters $(length(names))"
    return pdf(pd, NamedTuple{names}(x))
end

function logpdf(pd::ParameterDistribution, x::AbstractVector{<:Real})
    names = keys(pd)
    @assert length(x) == length(names) "Vector length $(length(x)) must match number of parameters $(length(names))"
    return logpdf(pd, NamedTuple{names}(x))
end

function _vectorized_rand(rng::Random.AbstractRNG, pd::ParameterDistribution)
    return collect(values(rand(rng, pd.inner)))
end

function _vectorized_rand(rng::Random.AbstractRNG, pd::ParameterDistribution, dims::Int...)
    samples = rand(rng, pd.inner, dims...)  # Array of NamedTuples with shape dims
    result = Array{Float64}(undef, length(keys(pd)), dims...)
    for idx in CartesianIndices(samples)
        nt = samples[idx]
        for (i, val) in enumerate(values(nt))
            result[i, Tuple(idx)...] = val
        end
    end
    return result
end

# Sampling delegates
Random.rand(rng::Random.AbstractRNG, pd::ParameterDistribution) = _vectorized_rand(rng, pd.inner)
Random.rand(rng::Random.AbstractRNG, pd::ParameterDistribution, dims::Int...) = _vectorized_rand(rng, pd.inner, dims...)
Base.rand(pd::ParameterDistribution) = _vectorized_rand(Random.default_rng(), pd)
Base.rand(pd::ParameterDistribution, dims::Int...) = _vectorized_rand(Random.default_rng(), pd, dims...)

# Broadcasting behavior
Base.Broadcast.broadcastable(pd::ParameterDistribution) = Ref(pd)

# Delegation for mean and covariance
import Distributions: mean, cov

mean(pd::ParameterDistribution) = mean(pd.inner)
cov(pd::ParameterDistribution) = Symmetric(Diagonal(collect(var(pd.inner))))

"""
    inverse(pd::ParameterDistribution, x::AbstractArray)

Apply the inverse transform for each parameter in a ParameterDistribution to an array `x` where the first axis corresponds to the parameters in `pd`.
Returns an array of the same shape as `x`, with the first axis transformed accordingly.
"""
function inverse(pd::ParameterDistribution, x::AbstractArray)
    dists = values(pd)
    nparams = length(dists)
    sz = size(x)
    @assert sz[1] == nparams "First axis of x must match number of parameters in pd"
    out = similar(x)
    trailing = ntuple(_ -> Colon(), ndims(x)-1)
    for i in 1:nparams
        dist = dists[i]
        sl = view(x, i, trailing...)
        if isa(dist, TransformedDistribution)
            out[i, trailing...] .= inverse.(dist.transform, sl)
        else
            out[i, trailing...] .= sl
        end
    end
    return out
end

"""
    (pd::ParameterDistribution)(x::AbstractArray)

Apply the forward transform for each parameter in a ParameterDistribution to an array `x` where the first axis corresponds to the parameters in `pd`.
Returns an array of the same shape as `x`, with the first axis transformed accordingly.
"""
function (pd::ParameterDistribution)(x::AbstractArray)
    dists = values(pd)
    nparams = length(dists)
    sz = size(x)
    @assert sz[1] == nparams "First axis of x must match number of parameters in pd"
    out = similar(x)
    trailing = ntuple(_ -> Colon(), ndims(x)-1)
    for i in 1:nparams
        dist = dists[i]
        sl = view(x, i, trailing...)
        if isa(dist, TransformedDistribution)
            out[i, trailing...] .= dist.transform.(sl)
        else
            out[i, trailing...] .= sl
        end
    end
    return out
end