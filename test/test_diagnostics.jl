using BayesJL
using Distributions, Random, LinearAlgebra

@testset "MCMC" begin
    
    @testset "Autocorrelation" begin

        rv = Normal(7.0, 3.0)

        M = 4
        N = 1_000 
        τ = 15.0

        Σ = var(rv) * exp.( -abs.(reshape(collect(1:N), N, 1) .- reshape(collect(1:N), 1, N)) ./ (0.5*τ))

        mvn = MvNormal(mean(rv) * ones(N), Symmetric(Σ))

        Random.seed!(42)
        chain = Array{Float64,3}(undef, 1, M, N)
        for m in 1:M
            chain[1, m, :] = rand(mvn)
        end

        ((μ, σ, τ, N̄, R̂),) = compute_diagnostics(chain)

        @test μ ≈ 6.890551958232697 atol=1e-12
        @test σ ≈ 3.0219616743777284 atol=1e-12
        @test τ ≈ 12.218312772709218 atol=1e-12
        @test N̄ ≈ 327.3774435480475 atol=1e-12
        @test R̂ ≈ 1.0105064718338395 atol=1e-12

    end

    @testset "Logprobabilities" begin

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

        # Analytical evidence
        z = 1/sqrt(2*pi*(σy^2 + σprior^2)) * exp( -0.5 * (μy - μprior)^2 / (σy^2 + σprior^2) )

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
        @test autocorrelation(importance_pdf) ≈ autocorrelation_exp(importance_logpdf) rtol=1e-13

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

    end

end