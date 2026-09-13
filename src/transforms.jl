"""
    transforms.jl

Provides transformation types and methods for mapping between bounded and unbounded spaces,
useful for MCMC sampling and variable transformations.
"""

using Distributions

# ============================================================================
# Transform Interface
# ============================================================================

"""
    Transform

Abstract base type for transformations between spaces.

All subtypes must implement:
- `(t::Transform)(x)`: Forward transformation
- `inverse(t::Transform, y)`: Inverse transformation
- `jacobian(t::Transform, x)`: Jacobian determinant dy/dx
"""
abstract type Transform end

"""
    (t::Transform)(x)

Apply the forward transformation to `x`.

Must be implemented for all Transform subtypes.
"""
(T::Transform)(x) = error("Forward transform not implemented for $(typeof(T))")

# Treat Transform objects as scalars in broadcasting (e.g., inverse.(t, ys))
Base.Broadcast.broadcastable(t::Transform) = Ref(t)

"""
    inverse(t::Transform, y)

Apply the inverse transformation to `y`, returning to the original space.

Must be implemented for all Transform subtypes.
"""
inverse(t::Transform, y) = error("Inverse transform not implemented for $(typeof(t))")

"""
    jacobian(t::Transform, x)

Compute the Jacobian determinant dy/dx at point `x`.

Must be implemented for all Transform subtypes.
"""
jacobian(t::Transform, x) = error("Jacobian not implemented for $(typeof(t))")

# ============================================================================
# LogTransform: (0, ∞) → (-∞, ∞)
# ============================================================================

"""
    LogTransform <: Transform

Logarithmic transformation mapping (0, ∞) to (-∞, ∞).

# Transformation
- Forward: y = log(x)
- Inverse: x = exp(y)
- Jacobian: dy/dx = 1/x

# Examples
```jldoctest
julia> using BayesJL

julia> t = LogTransform();

julia> y = t(2.0)
0.6931471805599453

julia> inverse(t, y)
2.0

julia> jacobian(t, 2.0)
0.5
```
"""
struct LogTransform <: Transform end

function (T::LogTransform)(x)
    @boundscheck x > 0 || throw(DomainError(x, "x must be positive for LogTransform"))
    return log(x)
end

function inverse(T::LogTransform, y)
    x = exp(y)
    @boundscheck x > 0 || throw(DomainError(x, "inverse(y=$(y)) resulted in x=$(x) which is not positive"))
    return x
end

function jacobian(T::LogTransform, x)
    @boundscheck x > 0 || throw(DomainError(x, "x must be positive for LogTransform"))
    return 1.0 / x
end

function logjacobian(T::LogTransform, x)
    @boundscheck x > 0 || throw(DomainError(x, "x must be positive for LogTransform"))
    return -log(x)
end

# ============================================================================
# BoundedTransform: [a, b] → (-∞, ∞)
# ============================================================================

"""
    BoundedTransform <: Transform

Logit-based transformation mapping a bounded interval [a, b] to (-∞, ∞).

# Fields
- `a::Float64`: Lower bound
- `b::Float64`: Upper bound

# Transformation
- Forward: y = log((x - a) / (b - x))
- Inverse: x = (a + b·exp(y)) / (1 + exp(y))
- Jacobian: dy/dx = (b - a) / ((x - a)(b - x))

# Constructors
```julia
BoundedTransform(a, b)        # Explicit bounds [a, b]
BoundedTransform()            # Default [0, 1] (standard logit)
BoundedTransform(d::Uniform)  # Extract bounds from Uniform distribution
```

# Examples
```jldoctest
julia> using BayesJL, Distributions

julia> t = BoundedTransform();

julia> t(0.5)  # Center of [0, 1] maps to 0
0.0

julia> inverse(t, 0.0)
0.5

julia> t = BoundedTransform(2.0, 5.0);

julia> y = t(3.5);  # Center of [2, 5]

julia> round(inverse(t, y), digits=10)
3.5

julia> d = Uniform(3.0, 7.0);

julia> t = BoundedTransform(d);

julia> t.a, t.b
(3.0, 7.0)

julia> t(5.0)
0.0
```
"""
struct BoundedTransform <: Transform
    a::Float64  # lower bound
    b::Float64  # upper bound
    tol::Float64 # tolerance for bounds checking

    function BoundedTransform(a::Real, b::Real; tol::Float64=1e-12)
        @assert a < b "Lower bound a must be less than upper bound b"
        new(Float64(a), Float64(b), tol)
    end
end

# Convenience constructors
BoundedTransform() = BoundedTransform(0.0, 1.0)
BoundedTransform(d::Uniform) = BoundedTransform(d.a, d.b)

function (t::BoundedTransform)(x)
    @boundscheck (x > t.a - t.tol) && (x < t.b + t.tol) || throw(DomainError(x, "x must be in [$(t.a), $(t.b)] for BoundedTransform"))
    return log((x - t.a) / (t.b - x))
end

function inverse(t::BoundedTransform, y)
    exp_y = exp(y)
    if exp_y == Inf
        return t.b
    elseif exp_y == 0.0
        return t.a
    else
        x = (t.a + t.b * exp_y) / (1 + exp_y)
        @boundscheck (x > t.a-t.tol) && (x < t.b+t.tol) || throw(DomainError(x, "inverse(y=$(y)) resulted in x=$(x) outside bounds ($(t.a), $(t.b))"))
        return x
    end
end

function jacobian(t::BoundedTransform, x)
    @boundscheck (x > t.a - t.tol) && (x < t.b + t.tol) || throw(DomainError(x, "x must be in ($(t.a), $(t.b)) for BoundedTransform"))
    jac = (t.b - t.a) / ((x - t.a) * (t.b - x))
    return jac
end

function logjacobian(t::BoundedTransform, x)
    @boundscheck (x > t.a - t.tol) && (x < t.b + t.tol) || throw(DomainError(x, "x must be in ($(t.a), $(t.b)) for BoundedTransform"))
    if isapprox(x, t.a; atol=t.tol) || isapprox(x, t.b; atol=t.tol)
        return Inf
    end
 
    logjac = log(t.b - t.a) - log(x - t.a) - log(t.b - x)
    return logjac
end