using BayesJL
using Distributions, Random

@testset "MCMC" begin
    
    @testset "Univariate sampling" begin

        # Define the target distribution
        params = ParameterDistribution(θ=
            TransformedDistribution(lognormal_from_moments(7.0, 3.0))
        )

        # MCMC configuration
        Nwalkers = 4
        Nsamples_per_walker = 1_000
        Nburnin = 100

        # Sample a starting point
        Random.seed!(42)
        y0 = sample_starting_points(params, Nwalkers)

        # Sample the target distribution
        lp = y -> logpdf(params, y)
        chain, lpvals = sample_distribution(lp, Nwalkers, Nsamples_per_walker, y0, Nburnin)

        # Compute evidence
        evidence, ρ, N̄ = evidence_inverse_importance(chain, lpvals)

        # Test evidence
        @test mean(evidence) ≈ 0.9974 atol=1e-4
        @test std(evidence) ≈ 0.0256 atol=1e-4
    end

    @testset "Plausibilities" begin

        Random.seed!(42)

        ẑ⁻¹ = ( lognormal_from_moments(100.0, 10.0), lognormal_from_moments(100.0, 10.0) )
        ρ = compute_plausibilities(ẑ⁻¹)

        @test mean(ρ[1]) ≈ 0.500551960837771 atol=1e-12
        @test mean(ρ[2]) ≈ 0.49946024721858234 atol=1e-12
        @test std(ρ[1]) ≈ 0.035119467906135196 atol=1e-12
        @test std(ρ[2]) ≈ 0.03508362360312623 atol=1e-12
    end
end