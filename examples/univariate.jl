using Plots, BayesJL, Distributions, LaTeXStrings

Nwalkers = 4
Nsamples_per_walker = 10_000
Nburnin = 1_000

function test_distribution(X, Y; y_has_mean=false, x_has_mean=false)
       
       # Define the target distribution
       parameters = ParameterDistribution(θ=Y)

       # Sample a starting point
       y0 = sample_starting_points(parameters, Nwalkers)

       # Sample the target distribution
       lp = y -> logpdf(parameters, y)
       chain, lpvals = sample_distribution(lp, Nwalkers, Nsamples_per_walker, y0, Nburnin)

       print_diagnostics(chain; param_names=keys(parameters))

       # Compute the marginalization
       unity, ρ, N̄ = evidence_inverse_importance(chain, lpvals)
       println("N̄ = ", N̄)
       println("evidence = ", mean(unity), " ± ", std(unity))

       # Post-processing
       flatchain, _ = flatten_chain(chain, lpvals)

       # Trace plots
       p_traces = plot(layout=(Nwalkers, 1), legend=false)
       for i in 1:Nwalkers
       plot!(p_traces[i], chain[1, i, :], 
              xlabel=(i == Nwalkers ? L"n" : ""),
              ylabel=L"y^{%$i}")
       end
       display(p_traces)

       # Histogram in the original (top) and transformed (bottom) domain
       y_samples = flatchain[1, :]
       x_samples = inverse.(Y.transform, y_samples)

       # Determine plot ranges from sampled values
       x_range = range(minimum(x_samples), maximum(x_samples), length=200)
       y_range = range(minimum(y_samples), maximum(y_samples), length=200)

       # Combined plot: original domain (top) and transformed (bottom) domain
       layout = plot(layout=(2, 1), size=(600, 400), legend=:topright)

       # Top: Original X domain
       histogram!(layout[1], x_samples, normalize=:pdf, alpha=0.5, label="MCMC", xlabel=L"x", ylabel=L"p(x)")
       plot!(layout[1], x_range, pdf.(X, x_range), linewidth=2, color=:red, label="True")
       if x_has_mean
              vline!(layout[1], [mean(X)], linewidth=2, linestyle=:dash, color=:black, label="True mean")
       end
       vline!(layout[1], [mean(x_samples)], linewidth=2, linestyle=:dot, color=:blue, label="Sample mean")
       # Bottom: Transformed Y domain
       histogram!(layout[2], flatchain[1, :], normalize=:pdf, alpha=0.5, label="MCMC", xlabel=L"y", ylabel=L"p(y)")
       plot!(layout[2], y_range, pdf.(Y, y_range), linewidth=2, color=:red, label="True")

       if y_has_mean
              vline!(layout[2], [mean(Y)], linewidth=2, linestyle=:dash, color=:black, label="True mean")
       end
       vline!(layout[2], [mean(y_samples)], linewidth=2, linestyle=:dot, color=:blue, label="Sample mean")

       display(layout)
end

# Log-normal distribution
X = lognormal_from_moments(7.0, 3.0)
Y = TransformedDistribution(X)

test_distribution(X, Y; x_has_mean=true, y_has_mean=true)

# Exponential distribution
X = Exponential(5.0)
Y = TransformedDistribution(X)

test_distribution(X, Y; x_has_mean=true, y_has_mean=true)

# Uniform distribution
X = Uniform(3.0, 7.0)
Y = TransformedDistribution(X)

test_distribution(X, Y; x_has_mean=true)

# LogitNormal distribution
X = LogitNormal(0.5, 1.5)
Y = TransformedDistribution(X)

test_distribution(X, Y)