using Plots, LaTeXStrings, LinearAlgebra
using BayesJL, Distributions, Random
using StatsPlots
using Statistics
using NearestNeighbors
using SpecialFunctions

Random.seed!(42)

# Define the synthetic data
gt  = (a=3.5, b=-0.75) # Ground truth coefficients
n   = 10 # Number of data points
x   = range(0.0, 4.0, length=n)
σy  = 0.3 # Noise level
ly  = 0.1 # Noise correlation length

Σy = zeros(n, n)
for i in 1:n
    for j in 1:n
        Σy[i, j] = σy^2 * exp(-abs(x[i] - x[j]) / ly)
    end
end

y = rand(MvNormal(gt.a .+ gt.b .* x, Σy))
scenario = (x = x, y = (y,))

# Plot the data
p_posterior = plot(scenario.x, scenario.y, marker=:circle, line=nothing,
                    size=(600, 400), legend=:topright, xlabel=L"x", ylabel=L"y", label="data", ylims=(0,4))

# ============================================================================
# Define the model
# ============================================================================

function linear_polynomial(;x, a=nothing, b=nothing)
    return a .+ b .* x
end
model = ModelClass(linear_polynomial)

# ============================================================================
# Setting up the inference problem
# ============================================================================

prior  = ParameterDistribution(a = Normal(gt.a, 0.2*abs(gt.a)), b = Normal(gt.b, 0.2*abs(gt.b)))
ε      = ProbabilisticError( ConstantCovariance(Σy) )

# ============================================================================
# Reference solution (conjugate prior)
# ============================================================================

H    = hcat(ones(length(x)), collect(x))
Σy⁻¹ = inv(Σy)

μ_prior = collect(mean(prior))
Σ_prior = cov(prior)
Σ_prior⁻¹ = inv(Σ_prior)

Σ_post = Symmetric(inv(Σ_prior⁻¹ + H' * Σy⁻¹ * H))
μ_post = Σ_post * (Σ_prior⁻¹ * μ_prior .+ H' * Σy⁻¹ * y)

# Construct MvNormal for prior and posterior
prior_mvn = MvNormal(μ_prior, Σ_prior)
post_mvn = MvNormal(μ_post, Σ_post)

# ============================================================================
# Sampling
# ============================================================================

Nwalkers = 2*length(prior)
Nsamples_per_walker = 50_000
Nburnin = 5_000
N = Nwalkers * Nsamples_per_walker

# Sample a starting point
y0 = sample_starting_points(prior, Nwalkers; Ncandidates=10*Nwalkers)

# Sample the posterior
chain, lpvals = sample_distribution(y -> logposterior(y, model, prior, (scenario,), ε),
                                     Nwalkers, Nsamples_per_walker, y0, Nburnin)

print_diagnostics(chain; param_names=keys(prior))

# Flattten the chain
flatchain, _ = flatten_chain(chain, lpvals)
N_params = size(flatchain, 1)
N_samples = size(flatchain, 2)

# Compare the sampled posterior to the reference solution
for j in 1:N_params
    println("mean($(keys(prior)[j])): $(mean(flatchain[j, :])) ($(μ_post[j]))")
    for k in j:N_params
        println("cov($(keys(prior)[j]), $(keys(prior)[k])): $(cov(flatchain[[j,k], :]'; corrected=true)[1,2]) ($(Σ_post[j,k]))")
    end
end

# ============================================================================
# KL divergence between sampled posterior and reference posterior
# ============================================================================

# Build a reference grid from sampled parameter ranges
N_ref = 50
param_min = [minimum(flatchain[j, :]) for j in 1:N_params]
param_max = [maximum(flatchain[j, :]) for j in 1:N_params]
param_grid = [range(param_min[j], param_max[j], length=N_ref) for j in 1:N_params]

# Compute k-th nearest-neighbor distance per grid point in scaled parameter space
# to avoid bias from different parameter units/magnitudes.
k_nn = 25

param_mean = vec(mean(flatchain, dims=2))
param_std  = vec(std(flatchain, dims=2))
@assert all(param_std .> 0.0) "At least one parameter has zero standard deviation; scaling is undefined."

flatchain_scaled = (flatchain .- param_mean) ./ param_std

grid_points = reduce(hcat, (collect(g) for g in Iterators.product(param_grid...)))
grid_points_scaled = (grid_points .- param_mean) ./ param_std

# Build KD-tree in scaled parameter space and query k nearest neighbors for all grid points.
tree = KDTree(flatchain_scaled)
_, dists = knn(tree, grid_points_scaled, k_nn, true)

# For matrix output (k x N_grid), the last row is the k-th nearest distance.
kth_nn_dist = ndims(dists) == 2 ? vec(dists[end, :]) : [last(di) for di in dists]

# k-NN density estimate at each grid point.
V_unit_ball = pi^(N_params / 2) / gamma(N_params / 2 + 1)
@assert all(kth_nn_dist .> 0.0) "k-th nearest-neighbor distance contains zeros; increase k_nn or inspect duplicate samples."

# Distances were computed in scaled coordinates, so map back to original
# parameter-space density via the diagonal scaling Jacobian.
pdf_knn_scaled = k_nn ./ (N_samples .* V_unit_ball .* (kth_nn_dist .^ N_params))
pdf_knn = pdf_knn_scaled ./ prod(param_std)

# Useful for 2-parameter visualization; keeps original grid orientation.
kth_nn_dist_grid = reshape(kth_nn_dist, length(param_grid[1]), length(param_grid[2]))
pdf_knn_grid = reshape(pdf_knn, length(param_grid[1]), length(param_grid[2]))

agrid = param_grid[1]
bgrid = param_grid[2]
postgrid = [pdf(post_mvn, [x, y]) for x in agrid, y in bgrid]

@assert N_params == 2 "Current KL integration block assumes a 2D parameter grid."
da = Float64(step(agrid))
db = Float64(step(bgrid))
# Trapezoidal-rule weights: boundary grid points get half weight per axis.
wa = ones(length(agrid)); wa[[1, end]] .*= 0.5
wb = ones(length(bgrid)); wb[[1, end]] .*= 0.5
posterior_mass_on_grid = da * db * sum(postgrid .* (wa * wb'))

log_p_over_q = log.(postgrid ./ pdf_knn_grid)
p_log_p_over_q = postgrid .* log_p_over_q
kl_divergence = da * db * sum(p_log_p_over_q .* (wa * wb'))

println("Integral of exact posterior over grid: $(posterior_mass_on_grid)")
println("KL divergence (integral of p log(p/q)): $(kl_divergence)")

# ============================================================================
# Post-processing
# ============================================================================

CI = 0.95

# Trace plots
p_traces = plot(layout=(size(flatchain, 1),1), legend=false)
for j in 1:size(flatchain, 1)
    for i in 1:Nwalkers
        plot!(p_traces[j], chain[j, i, :], label="Walker $(i)", 
              xlabel=(j == size(flatchain, 1) ? L"n" : ""), ylabel="$(keys(prior)[j])",legend=:outerright)
    end
end
display(p_traces)

# Histograms of posterior parameters in original space
p_hist = plot(layout=(length(prior), 1), size=(600, 200*length(prior)), legend=false)
for (j, (param_name, param_dist)) in enumerate(pairs(prior))
    samples = flatchain[j, :]

    histogram!(p_hist, samples, subplot=j, xlabel=string(param_name), ylabel="",
               bins=50, normalize=:pdf, alpha=0.7, color=:blue)
    
    vline!(p_hist, [median(samples)], subplot=j, color=:red, linewidth=2, linestyle=:dash, label="Median")
    vline!(p_hist, [gt[param_name]], subplot=j, color=:black, linewidth=2, linestyle=:solid, label="Ground truth")
end
plot!(p_hist, subplot=1, legend=:topright)
display(p_hist)

# Collect posterior predictive samples
y_samples = zeros(length(scenario.x), size(flatchain, 2))
for i in 1:size(flatchain, 2)
    params = NamedTuple{keys(prior)}(flatchain[:, i])
    
    d = model(;scenario..., params...)
    
    # Sample from error model and add to prediction
    # Pass both scenario and parameters to allow functional covariances
    y_samples[:, i] = d .+ rand(ε; scenario..., params...)
end

# Compute credibility interval
α = (1 - CI) / 2
y_median = vec(median(y_samples, dims=2))
y_lower = vec(mapslices(x -> quantile(x, α), y_samples, dims=2))
y_upper = vec(mapslices(x -> quantile(x, 1-α), y_samples, dims=2))

# Plot credibility interval band on both subplots
plot!(p_posterior, scenario.x, y_median, ribbon=(y_median .- y_lower, y_upper .- y_median),
       fillalpha=0.3, label="$(Int(CI*100))% CI", color=:blue, linewidth=2)

display(p_posterior)

# 2D histogram (heatmap) of posterior samples for parameters a and b
p_hist = histogram2d(flatchain[1, :], flatchain[2, :], (10, 10); normalize=:pdf, xlabel="a", ylabel="b", 
            colorbar=true, size=(500,400), show_empty_bins=true, c=:heat)

contour!(p_hist, agrid, bgrid, postgrid', fill=false, linewidth=1, levels=5, clabels=true, c=:black)

p_log_ratio = heatmap(agrid, bgrid, log_p_over_q';
                      xlabel="a", ylabel="b", colorbar=true, colorbar_title="log(p/q)",
                      c=:heat, size=(500, 400))

p_weighted_log_ratio = heatmap(agrid, bgrid, p_log_p_over_q';
                               xlabel="a", ylabel="b", colorbar=true, colorbar_title="p log(p/q)",
                               c=:heat, size=(500, 400))

plot(p_hist, xlims=(μ_post[1]-2sqrt(Σ_post[1,1]), μ_post[1]+2sqrt(Σ_post[1,1])),
          ylims=(μ_post[2]-2sqrt(Σ_post[2,2]), μ_post[2]+2sqrt(Σ_post[2,2])))

display(p_hist)
display(p_log_ratio)
display(p_weighted_log_ratio)