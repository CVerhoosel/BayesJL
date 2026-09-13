using Test
using Documenter
using BayesJL
import Logging

@testset "Unittests" begin
    include("test_distributions.jl")
    include("test_mcmc.jl")
    include("test_diagnostics.jl")
end

@testset "Doctests" begin
    # Suppress Documenter warnings during doctest
    Logging.with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        doctest(BayesJL)
    end
end