### A Pluto.jl notebook ###
# v0.20.28

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 2a8d3f6c-8e1a-4f5b-b9d2-7c6e1a4f0b3d
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "Project.toml"))
    Pkg.instantiate()
end

# ╔═╡ 4cce3a34-8505-4c19-a0d8-b51d63b6e68a
begin
    using BayesJL
    using Distributions
    using LaTeXStrings
    using Plots
    using PlutoUI
    using Random
    using Statistics
end

# ╔═╡ 8b2f5b10-4efb-4a8a-a2bf-4f06a9d3e1f1
md"""
# Univariate transformed distributions

This notebook reproduces the univariate example for BayesJL. Select a distribution to compare its original-space density with MCMC samples in transformed space.
"""

# ╔═╡ 1cfe1e88-4f04-4c74-9b65-a4c5e08d8d5c
md"""
## Choose a distribution

The selected distribution is transformed to an unconstrained parameter space before sampling.
"""

# ╔═╡ 0b1b7ddf-6f83-4450-85b8-47a5d6d3e1cc
@bind distribution_name Select([
    "Log-normal",
    "Exponential",
    "Uniform",
    "Logit-normal",
])

# ╔═╡ 5d70ad77-1f1e-4cb7-bc32-b7f15cde4a6c
begin
    distribution_specs = Dict(
        "Log-normal" => (original = lognormal_from_moments(7.0, 3.0),
                          has_original_mean = true,
                          has_transformed_mean = true),
        "Exponential" => (original = Exponential(5.0),
                          has_original_mean = true,
                          has_transformed_mean = true),
        "Uniform" => (original = Uniform(3.0, 7.0),
                      has_original_mean = true,
                      has_transformed_mean = false),
        "Logit-normal" => (original = LogitNormal(0.5, 1.5),
                           has_original_mean = false,
                           has_transformed_mean = false),
    )

    selected = distribution_specs[distribution_name]
    X = selected.original
    Y = TransformedDistribution(X)
end

# ╔═╡ 6d4f1e4b-37e6-4d97-a5f8-cd2e5b0a8d6d
md"""
The original distribution is **$(distribution_name)**. BayesJL samples the transformed distribution `Y`, then maps the samples back to the original variable `X` for comparison.
"""

# ╔═╡ 6bd63ca3-3c70-4c18-a4c6-e2f9a3d8f7a2
md"""
## Sampling settings

The defaults reproduce the original example. The slider bounds keep the maximum
interactive run to 84,000 sampling steps across all walkers.
"""

# ╔═╡ 7e1a4c9f-2b6d-4f80-a3e5-9c1d7b8f0a2e
@bind Nwalkers Slider(2:2:12; default=4, show_value=true)

# ╔═╡ 9f3b7d1c-5a8e-4c20-b6f2-0d9e1a4c7b8f
@bind Nsamples_per_walker Slider(250:250:5_000; default=2_000, show_value=true)

# ╔═╡ 4a8c2e6f-1b5d-4f90-a7c3-8e0d2b6f9a1c
@bind Nburnin Slider(100:100:2_000; default=500, show_value=true)

# ╔═╡ 9838ff0e-4c5a-4ff5-bf2d-1c9f34d2c6b8
begin
    Random.seed!(42)

    parameters = ParameterDistribution(θ = Y)
    y0 = sample_starting_points(parameters, Nwalkers)
    log_probability = y -> logpdf(parameters, y)
    chain, lpvals = sample_distribution(
        log_probability,
        Nwalkers,
        Nsamples_per_walker,
        y0,
        Nburnin,
    )
    flatchain, _ = flatten_chain(chain, lpvals)
end

# ╔═╡ 2f7bbbc4-98f4-4bb1-a0a0-7a1c3d5e9f62
begin
    inverse_evidence, fraction, effective_samples = evidence_inverse_importance(chain, lpvals)
    diagnostics = print_diagnostics(chain; param_names=keys(parameters))

    md"""
    **Inverse evidence estimate:** $(round(mean(inverse_evidence), digits=4)) ± $(round(std(inverse_evidence), digits=4))  
    **Effective samples:** $(effective_samples)  
    **Samples inside importance region:** $(round(fraction, digits=3))
    """
end

# ╔═╡ d6e20b1c-1b42-4f85-9a7a-7f4d2c8e3b5f
begin
    y_samples = vec(flatchain[1, :])
    x_samples = inverse.(Y.transform, y_samples)

    x_range = range(minimum(x_samples), maximum(x_samples), length=200)
    y_range = range(minimum(y_samples), maximum(y_samples), length=200)

    density_plot = plot(
        layout=(2, 1),
        size=(700, 500),
        legend=:topright,
    )

    histogram!(
        density_plot[1],
        x_samples,
        normalize=:pdf,
        alpha=0.5,
        label="MCMC",
        xlabel=L"x",
        ylabel=L"p(x)",
    )
    plot!(
        density_plot[1],
        x_range,
        pdf.(X, x_range),
        linewidth=2,
        color=:red,
        label="True",
    )
    if selected.has_original_mean
        vline!(
            density_plot[1],
            [mean(X)],
            linewidth=2,
            linestyle=:dash,
            color=:black,
            label="True mean",
        )
    end
    vline!(
        density_plot[1],
        [mean(x_samples)],
        linewidth=2,
        linestyle=:dot,
        color=:blue,
        label="Sample mean",
    )

    histogram!(
        density_plot[2],
        y_samples,
        normalize=:pdf,
        alpha=0.5,
        label="MCMC",
        xlabel=L"y",
        ylabel=L"p(y)",
    )
    plot!(
        density_plot[2],
        y_range,
        pdf.(Y, y_range),
        linewidth=2,
        color=:red,
        label="True",
    )
    if selected.has_transformed_mean
        vline!(
            density_plot[2],
            [mean(Y)],
            linewidth=2,
            linestyle=:dash,
            color=:black,
            label="True mean",
        )
    end
    vline!(
        density_plot[2],
        [mean(y_samples)],
        linewidth=2,
        linestyle=:dot,
        color=:blue,
        label="Sample mean",
    )

    density_plot
end

# ╔═╡ Cell order:
# ╟─8b2f5b10-4efb-4a8a-a2bf-4f06a9d3e1f1
# ╟─2a8d3f6c-8e1a-4f5b-b9d2-7c6e1a4f0b3d
# ╟─4cce3a34-8505-4c19-a0d8-b51d63b6e68a
# ╟─1cfe1e88-4f04-4c74-9b65-a4c5e08d8d5c
# ╟─0b1b7ddf-6f83-4450-85b8-47a5d6d3e1cc
# ╟─5d70ad77-1f1e-4cb7-bc32-b7f15cde4a6c
# ╟─6d4f1e4b-37e6-4d97-a5f8-cd2e5b0a8d6d
# ╠═6bd63ca3-3c70-4c18-a4c6-e2f9a3d8f7a2
# ╠═7e1a4c9f-2b6d-4f80-a3e5-9c1d7b8f0a2e
# ╠═9f3b7d1c-5a8e-4c20-b6f2-0d9e1a4c7b8f
# ╠═4a8c2e6f-1b5d-4f90-a7c3-8e0d2b6f9a1c
# ╟─9838ff0e-4c5a-4ff5-bf2d-1c9f34d2c6b8
# ╟─2f7bbbc4-98f4-4bb1-a0a0-7a1c3d5e9f62
# ╟─d6e20b1c-1b42-4f85-9a7a-7f4d2c8e3b5f
