# Standard HAR(1, 5, 22) model. 
using BFlux, Flux
using Distributions, Random
using CSV, DataFrames
using StatsPlots
using MCMCChains
using LinearAlgebra
using Random
using Serialization

Random.seed!(6150533)

# ------------------------------------------------------------------------------
# Project Includes
# 
# Notes: 
# - We will use BFlux with a single linear layer and FeedforwardNormal
#   likelihood and thus do not need to include the HARLikelihood
# ------------------------------------------------------------------------------
include("./dataloader.jl")

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

import Base: copy, deepcopy
import BFlux: MCMCState

copy(s::T) where {T<:MCMCState} = T([getfield(s, k) for k in fieldnames(T)]...)
deepcopy(s::T) where {T<:MCMCState} = T([Base.deepcopy(getfield(s, k)) for k in fieldnames(T)]...)

function parallel_mcmc(nchains, bnn, batchsize, nsamples, sampler; kwargs...)
    samplers = [deepcopy(sampler) for _ in 1:nchains]
    T = eltype(bnn.like.nc.θ)
    chains = Array{T, 3}(undef, bnn.num_total_params, nsamples, nchains)
    Threads.@threads for i=1:nchains
        chains[:,:,i] = mcmc(bnn, batchsize, nsamples, samplers[i]; kwargs..., showprogress = false)
    end
    return chains, samplers
end

function naive_prediction(bnn, draw, x)
    T = eltype(draw)
    θnet, θhyper, θlike = split_params(bnn, draw)
    net = bnn.like.nc(T.(θnet))
    return vec(net(x))
end

function naive_prediction_mcmc(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    coeffs = Array{T, 3}(undef, 4, length(y), size(draws,2))
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    lps = Array{T, 1}(undef, size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        lps[i] = loglikeprior(bnn, draws[:,i], x, y)
        net = bnn.like.nc(draws[:, i])
        yh = vec(net(x))
        yhats[:,i] = yh
    end
    return yhats, lps
end

function plot_quantile_comparison(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    plot(target_q, observed_q, label = "Observed", legend_position = :topleft, 
        xlab = "Quantile of Posterior Draws", 
        ylab = "Percent Observations below"
    )
    plot!(x -> x, minimum(target_q), maximum(target_q), label = "Theoretical")
end

# ------------------------------------------------------------------------------
# Data
# 
# Notes: 
# - We will use the standard HAR model and thus do only need to rv_matrix 
# - Split this into train, validate, test
# - No scaling will be used here since it is not needed in linear regressions
# - variables are already in logs
# ------------------------------------------------------------------------------

df = load_data()
rv_matrix, dates = get_rv_matrix(df)
# We want to predict one day ahead
x = permutedims(rv_matrix[1:end-1,:], [2, 1])
y = rv_matrix[2:end, 1]
train, validate, test = split_train_validate_test(x, y, dates[2:end])

# ------------------------------------------------------------------------------
# Model
# 
# Notes: 
# - Since we are lazy and we already have BFlux to do flexible Bayesian
#   analysis, we will implement linear regression as a single linear layer NN.
# - From Corsi(2009) we would expext the coefficient to be somewhere between
#   zero and 1, or even more precise somewhere between 0 and 0.5. We will
#   therefore use a Gaussian prior with mean zero and standard deviation 0.5. 
# - We will use a Gaussian likelihood (as an alternative we could use a
#   T-likelihood). For the standard deviation we will use a Gamma(2.0, 0.5)
#   prior which has a mean of 1.0 and has 90% of its probability mass between
#   0.18 and 1.94. This is probably a much too wide prior for this application,
#   but that is better than artifically enforcing too narror priors.
# - For all methods that need an initial value, we will use a Normal(0, 0.5)
#   initialisation. This is the same as the prior on the coefficient here since
#   no hidden state network parameters exist.
# ------------------------------------------------------------------------------
model = Flux.Chain(Dense(3, 1))
nc = destruct(model)
prior = GaussianPrior(nc, 0.5f0)
like = FeedforwardNormal(nc, Gamma(2.0, 0.5))
init = InitialiseAllSame(Normal(0.0f0, 0.5f0), like, prior)
bnn = BNN(train.x, train.y, like, prior, init)

# ------------------------------------------------------------------------------
# MAP Estimate
# 
# Notes: 
# - This is mostly used to make sure that the model is correctly specified and
#   that we are able to make good predictions. If the MAP estimate produces bad
#   results, then something would be wrong in the specification.
# ------------------------------------------------------------------------------
Random.seed!(6150533)
opt = FluxModeFinder(bnn, Flux.ADAM())
βmap = find_mode(bnn, 100, 1000, opt)
serialize("./outputs/har-map-beta.jld", βmap)

# Train data
yhat_map = naive_prediction(bnn, βmap, train.x)
p = plot(train.dates, train.y; color = :black, label = "Train Data");
p = plot!(p, train.dates, yhat_map; color = :red, linewidth = 2, label = "MAP")
savefig(p, "./outputs/har-compare-train.pdf")

rmse = sqrt(mean(abs2, bnn.y - yhat_map))
serialize("./outputs/har-rmse-train.jld", rmse)

# Validation data
yhat_map = naive_prediction(bnn, βmap, validate.x)
p = plot(validate.dates, validate.y; color = :black, label = "Train Data");
p = plot!(p, validate.dates, yhat_map; color = :red, linewidth = 2, label = "MAP")
savefig(p, "./outputs/har-compare-validation.pdf")

rmse = sqrt(mean(abs2, validate.y - yhat_map))
serialize("./outputs/har-rmse-validation.jld", rmse)

# Test data
yhat_map = naive_prediction(bnn, βmap, test.x)
p = plot(test.dates, test.y; color = :black, label = "Test Data");
p = plot!(p, test.dates, yhat_map; color = :red, linewidth = 2, label = "MAP")
savefig(p, "./outputs/har-compare-test.pdf")

rmse = sqrt(mean(abs2, test.y - yhat_map))
serialize("./outputs/har-rmse-test.jld", rmse)

# ------------------------------------------------------------------------------
# MCMC Estimation
# 
# Notes: 
# - AdaptiveMH was chosen because all gradient based methods produced very poor
#   chains. I also tried to implement it in Turing.jl using NUTS but that took
#   ages. This probably has to do with the high correlation in the variables and
#   the long time series.
# ------------------------------------------------------------------------------

Random.seed!(6150533)
sampler = AdaptiveMH(diagm(one.(βmap)), 1000, 3f-1, 1f-5)
chs, samplers = parallel_mcmc(4, bnn, length(bnn.y), 50_000, sampler)
serialize("./outputs/har-ahmc-sampler.jld", sampler)
serialize("./outputs/har-ahmc-chain.jld", chs)

chs = chs[:, 5001:end, :]
mean_acceptance_rates = [mean(s.accepted[5001:end]) for s in samplers]
serialize("./outputs/har-ahmc-mean-accept.jld", mean_acceptance_rates)

chains = Chains(permutedims(chs, [2, 1, 3]), [:intercept, :daily, :weekly, :monthly, :logσ])
p = plot(chains)
savefig(p, "./outputs/har-ahmc-chain-plot.pdf")

ch = reduce(hcat, eachslice(chs; dims = 3))
ch = ch[:, 1:16:end]

# Train data
yhat, lps = naive_prediction_mcmc(bnn, ch; x = train.x, y = train.y)
yh = mean(yhat; dims = 2)
p = plot(train.dates, train.y; color = :black, label = "Train Data");
p = plot!(p, train.dates, yh; color = :red, linewidth = 2, label = "AMH")
savefig(p, "./outputs/har-ahmc-compare-train.pdf")

rmse = sqrt(mean(abs2, bnn.y - yh))
serialize("./outputs/har-ahmc-rmse-train.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = train.x)
p = plot_quantile_comparison(train.y, posterior_yhat)
savefig(p, "./outputs/har-ahmc-qqplot-train.pdf")

# Validation data
yhat, lps = naive_prediction_mcmc(bnn, ch; x = validate.x, y = validate.y)
yh = mean(yhat; dims = 2)
p = plot(validate.dates, validate.y; color = :black, label = "Validation Data");
p = plot!(p, validate.dates, yh; color = :red, linewidth = 2, label = "AMH")
savefig(p, "./outputs/har-ahmc-compare-validation.pdf")

rmse = sqrt(mean(abs2, validate.y - yh))
serialize("./outputs/har-ahmc-rmse-validation.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = validate.x)
p = plot_quantile_comparison(validate.y, posterior_yhat)
savefig(p, "./outputs/har-ahmc-qqplot-validation.pdf")

# Test data
yhat, lps = naive_prediction_mcmc(bnn, ch; x = test.x, y = test.y)
yh = mean(yhat; dims = 2)
p = plot(test.dates, test.y; color = :black, label = "Test Data");
p = plot!(p, test.dates, yh; color = :red, linewidth = 2, label = "AMH")
savefig(p, "./outputs/har-ahmc-compare-test.pdf")

rmse = sqrt(mean(abs2, test.y - yh))
serialize("./outputs/har-ahmc-rmse-test.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = test.x)
p = plot_quantile_comparison(test.y, posterior_yhat)
savefig(p, "./outputs/har-ahmc-qqplot-test.pdf")