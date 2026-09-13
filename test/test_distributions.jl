using BayesJL
using Distributions, Random

@testset "Distributions" begin
    
    @testset "lognormal_from_moments" begin
        d = lognormal_from_moments(7.0, 3.0)
        @test mean(d) ≈ 7.0 atol=1e-12
        @test std(d) ≈ 3.0 atol=1e-12
        @test var(d) ≈ 9.0 atol=1e-12
        @test meanlogx(d) ≈ 1.8615987928 atol=1e-9
        @test stdlogx(d) ≈ 0.4106369594 atol=1e-9
    end
    
    @testset "TransformedDistribution" begin

        for d in [lognormal_from_moments(7.0, 3.0), LogitNormal(0.5, 1.5)]
            
            td = TransformedDistribution(d)

            μ, σ = params(d)
            
            n = Normal(μ, σ)
            
            @testset "(log)pdf" begin
                y_rng = range(μ - 3σ, μ + 3σ, length=100)
                Δy = step(y_rng)

                y = collect(y_rng)
                pdf_td = pdf.(td, y)
                pdf_n = pdf.(n, y)
                
                @test Δy * sum(abs.(pdf_td .- pdf_n)) < 1e-12
                        
                @test pdf(td, μ) ≈ pdf(n, μ) atol=1e-12
                @test pdf(td, μ + σ) ≈ pdf(n, μ + σ) atol=1e-12
                @test pdf(td, μ - σ) ≈ pdf(n, μ - σ) atol=1e-12

                @test logpdf(td, μ) ≈ logpdf(n, μ) atol=1e-12
                @test logpdf(td, μ + σ) ≈ logpdf(n, μ + σ) atol=1e-12
                @test logpdf(td, μ - σ) ≈ logpdf(n, μ - σ) atol=1e-12
            end

            @testset "rand" begin
                samples = rand(MersenneTwister(42), td, 20_000)
                @test mean(samples) ≈ μ rtol=5e-2
                @test std(samples) ≈ σ rtol=5e-2
            end

        end

        @testset "Log-transformed Exponential" begin
            θ = 2.5  # scale parameter
            d_exp = Exponential(θ)
            td_exp = TransformedDistribution(d_exp)

            # Theoretical moments for log(X) where X ~ Exponential(θ)
            μ_th = log(θ) - Base.MathConstants.eulergamma
            var_th = (pi^2)/6
            std_th = pi/sqrt(6)

            @test mean(td_exp) ≈ μ_th atol=1e-12
            @test var(td_exp) ≈ var_th atol=1e-12
            @test std(td_exp) ≈ std_th atol=1e-12
        end

        @testset "Parameter Distribution" begin
            a = Normal(7.0, 3.0)
            b = Normal(5.0, 1.0)

            pd = ParameterDistribution(a = a, b = b);

            @test all(isapprox.((mean(a), mean(b)), values(mean(pd)), rtol=1e-18))
            @test all(isapprox.([var(a) 0;0 var(b)], cov(pd), rtol=1e-18, atol=1e-18))

            Random.seed!(42)
            samples = rand(pd, 10)

            @test all(isapprox.( [pdf(pd, (a=s[1], b=s[2])) for s in eachcol(samples)], [pdf(a, s[1])*pdf(b, s[2]) for s in eachcol(samples)], rtol=1e-15 ))
            @test all(isapprox.( [logpdf(pd, (a=s[1], b=s[2])) for s in eachcol(samples)], [logpdf(a, s[1]) + logpdf(b, s[2]) for s in eachcol(samples)], rtol=1e-15 ))

            @test all(isapprox.(pd(inverse(pd, samples)), samples, rtol=1e-15))
        end
    end

end