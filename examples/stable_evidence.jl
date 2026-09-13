using BayesJL, Distributions, Plots, Random, LinearAlgebra, Test, Printf

# Datapoint
μy = 5.0
σy = 0.5
e  = Normal(0, σy)

# Prior
μprior = 2.0
σprior = 0.2
prior  = Normal(μprior, σprior)

# Posterior
σpost = 1 / sqrt(1/σprior^2 + 1/σy^2)
μpost = σpost^2 * (μprior/σprior^2 + μy/σy^2)
post  = Normal(μpost, σpost)

θrng = collect(range(0, 9, length=1000))
fig = plot(θrng, pdf.(prior, θrng), label="Prior", xlabel="θ", ylabel="Density", legend=:topright)
plot!(fig, θrng, pdf.(e, θrng .- μy), label="Likelihood")
plot!(fig, θrng, pdf.(post, θrng), label="Posterior")
display(fig)

# Analytical evidence
z = 1/sqrt(2*pi*(σy^2 + σprior^2)) * exp( -0.5 * (μy - μprior)^2 / (σy^2 + σprior^2) )

fig = plot(θrng, pdf.(prior, θrng) .* pdf.(e, θrng .- μy) ./ pdf.(post, θrng), label="Evidence", xlabel="θ", ylabel="Density", legend=:topright)
plot!(fig, θrng , fill(z, length(θrng)), label="Analytical evidence")
display(fig)

# ============================================================================
# Generate correlated posterior samples
# ============================================================================

M = 2
N = 1_000 
τ = 15.0

Random.seed!(42)

Σ = var(post) * exp.( -abs.(reshape(collect(1:N), N, 1) .- reshape(collect(1:N), 1, N)) ./ (0.5*τ))

mvn = MvNormal(mean(post) * ones(N), Symmetric(Σ))

chain = Array{Float64,3}(undef, 1, M, N)
for m in 1:M
    chain[1, m, :] = rand(mvn)
end

# ============================================================================
# Evidence calculation
# ============================================================================

# Importance sampling
lφ, ρ = BayesJL._ellipsoid_logpdf(chain)
φ = exp.(lφ)

prior_pdf = pdf.(prior, chain)[1,:,:]
likelihood_pdf = pdf.(e, chain .- μy)[1,:,:]
unnormalized_posterior_pdf = prior_pdf .* likelihood_pdf
importance_pdf = φ ./ unnormalized_posterior_pdf

ẑ⁻¹ = mean(importance_pdf)
varẑ⁻¹ = var(importance_pdf)

prior_logpdf = logpdf.(prior, chain)[1,:,:]
likelihood_logpdf = logpdf.(e, chain .- μy)[1,:,:]
unnormalized_posterior_logpdf = prior_logpdf .+ likelihood_logpdf
importance_logpdf = lφ .- unnormalized_posterior_logpdf

lẑ⁻¹ = logmeanexp(importance_logpdf)
lvarẑ⁻¹ = logvarexp(importance_logpdf)

@test lẑ⁻¹ ≈ log(ẑ⁻¹) rtol=1e-15
@test lvarẑ⁻¹ ≈ log(varẑ⁻¹) rtol=1e-14

println(@sprintf("Evidence estimate: %g (error: %4.3f%%)", 1 / ẑ⁻¹, (1 / ẑ⁻¹ - z) / z * 100))
println(@sprintf("Evidence estimate (log): %g (error: %4.3f%%)", 1 / exp(lẑ⁻¹), (1 / exp(lẑ⁻¹) - z) / z * 100))

@test 1.0/ẑ⁻¹ ≈ 1.32975e-07 rtol=1e-5
@test 1.0/exp(lẑ⁻¹) ≈ 1.32975e-07 rtol=1e-5

@test (1 / ẑ⁻¹ - z) / z * 100 ≈ -1.574 rtol=1e-3
@test (1 / exp(lẑ⁻¹) - z) / z * 100 ≈ -1.574 rtol=1e-3

# Pooled variance estimate
var⁺ = pooledvar( importance_pdf )
lvar⁺ = logpooledvarexp( importance_logpdf )
@test lvar⁺ ≈ log(var⁺) rtol=1e-15

# Variogram
@test log.(vario(chain)) ≈ logvarioexp(log.(chain)) rtol=1e-15
@test log.(vario(importance_pdf)) ≈ logvarioexp(importance_logpdf) rtol=1e-15

# Autocorrelation
@test autocorrelation(chain) ≈ autocorrelation_exp(log.(chain)) rtol=1e-12
@test autocorrelation(importance_pdf) ≈ autocorrelation_exp(importance_logpdf) rtol=1e-14

# Autocorrelation time
τ = autocorrelation_time(chain)
@test τ ≈ [13.971021526924284] rtol=1e-12
@test τ ≈ autocorrelation_time_exp(log.(chain)) rtol=1e-12

τ = autocorrelation_time(importance_pdf)
@test τ ≈ 3.5644714261668646 rtol=1e-12
@test τ ≈ autocorrelation_time_exp(importance_logpdf) rtol=1e-12

# Effective sample size
N̄ = effective_sample_size(chain)
@test N̄ ≈ [143.15345489559914] rtol=1e-12
@test N̄ ≈ effective_sample_size_exp(log.(chain)) rtol=1e-12

N̄ = effective_sample_size(importance_pdf)
@test N̄ ≈ 561.0930095603952 rtol=1e-12
@test N̄ ≈ effective_sample_size_exp(importance_logpdf) rtol=1e-12

# Inverse evidence
ẑ⁻¹, f, N̄ = evidence_inverse_importance(chain, unnormalized_posterior_logpdf)
l = lognormal_from_moments(mean(importance_pdf), sqrt(var(importance_pdf)/N̄) )

@test meanlogx(ẑ⁻¹) ≈ meanlogx(l) rtol=1e-12
@test stdlogx(ẑ⁻¹) ≈ stdlogx(l) rtol=1e-12