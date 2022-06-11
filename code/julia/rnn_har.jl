# ------------------------------------------------------------------------------
# RNN-HAR applied to SPY
# 
# Notes: 
# ------------------------------------------------------------------------------
using BFlux, Flux
using Distributions, Random
using CSV, DataFrames
using StatsPlots
using MCMCChains
using LinearAlgebra
using Random
using Serialization
using StatsBase

Random.seed!(6150533)

# ------------------------------------------------------------------------------
# Project Includes
# 
# Notes: 
# ------------------------------------------------------------------------------
include("./dataloader.jl")
include("./har_likelihood.jl")

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

function naive_prediction(bnn, draws::Array{T, 2}; x = bnn.x, y = bnn.y) where {T}
    coeffs = Array{T, 3}(undef, 4, length(y), size(draws,2))
    yhats = Array{T, 2}(undef, length(y), size(draws, 2))
    lps = Array{T, 1}(undef, size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        lps[i] = loglikeprior(bnn, draws[:,i], x, y)
        net = bnn.like.nc(draws[:, i])
        c = [net(xx) for xx in eachslice(x[1:end-1,1:3,:]; dims = 1)][end]
        xhar = x[end, :, :]
        yh = c[1,:] .+ c[2,:].*xhar[1,:] .+ c[3,:].*xhar[2,:] .+ c[4,:].*xhar[3,:]
        coeffs[:,:,i] = c
        yhats[:,i] = yh
    end
    return yhats, coeffs, lps
end

function get_observed_quantiles(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    qs = [quantile(yr, target_q) for yr in eachrow(posterior_yhat)]
    qs = reduce(hcat, qs)
    observed_q = mean(reshape(y, 1, :) .< qs; dims = 2)
    return observed_q
end

function plot_quantile_comparison(y, posterior_yhat, target_q = 0.05:0.05:0.95)
    observed_q = get_observed_quantiles(y, posterior_yhat, target_q)
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
# ------------------------------------------------------------------------------

x, y, dates = get_x_y_dates(;subsequence_length = 22)
train, validate, test = split_train_validate_test(x, y, dates)

# ------------------------------------------------------------------------------
# RNN-HAR MAP
# 
# Notes:
# ------------------------------------------------------------------------------
Random.seed!(6150533)

function map_estimate_net(train, validate, net)
    nc = destruct(net)
    prior = GaussianPrior(nc, 0.5f0)
    like = HARLikelihood(nc, Gamma(2.0, 0.5))
    init = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)
    bnn = BNN(train.x, train.y, like, prior, init)

    opt = FluxModeFinder(bnn, Flux.ADAM())
    θmap = find_mode(bnn, 100, 1000, opt)

    yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1))
    yh = mean(yhat; dims = 2)
    train_rmse = sqrt(mean(abs2, bnn.y .- yh))


    yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1); x = validate.x, y = validate.y)
    yh = mean(yhat; dims = 2)
    test_rmse = sqrt(mean(abs2, validate.y .- yh))
    return [train_rmse, test_rmse]
end

nets_to_test = [
    Flux.Chain(RNN(3, 10, tanh), Dense(10, 4)), 
    Flux.Chain(RNN(3, 3, tanh), Dense(3, 4)), 
    Flux.Chain(RNN(3, 20, tanh), Dense(20, 4)), 
    Flux.Chain(RNN(3, 10, tanh), RNN(10, 4, tanh), Dense(4, 4)),
    Flux.Chain(RNN(3, 10, tanh), Dense(10, 10, tanh), Dense(10, 4)), 
    Flux.Chain(Dense(3, 4, tanh), RNN(4, 4, tanh), Dense(4, 4)),
]

Random.seed!(6150533)
rmse = [map_estimate_net(train, validate, nn) for nn in nets_to_test]
rmse = reduce(hcat, rmse)

serialize("./outputs/rnn-har-map-networks-tested.jld", nets_to_test)
serialize("./outputs/rnn-har-map-network-rmses.jld", rmse)

# This is the network that we settled for
net = Flux.Chain(RNN(3, 3, tanh), Dense(3, 4))
nc = destruct(net)
prior = GaussianPrior(nc, 0.5f0)
like = HARLikelihood(nc, Gamma(2.0, 0.5))
init = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)
bnn = BNN(train.x, train.y, like, prior, init)

# MAP Estimate 

Random.seed!(6150533)
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 100, 500, opt)
serialize("./outputs/rnn-har-optimiser.jld", opt)
serialize("./outputs/rnn-har-map-estimate.jld", θmap)

yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1))
yh = mean(yhat; dims = 2)
p = plot(train.dates, bnn.y; color = :black, label = "Data");
p = plot!(p, train.dates, yh; color = :red, label = "MAP")
savefig(p, "./outputs/rnn-har-map-comparison.pdf")
rmse = sqrt(mean(abs2, bnn.y .- yh))
serialize("./outputs/rnn-har-map-train-rmse.jld", rmse)

# Validation

yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1); x = validate.x, y = validate.y)
yh = mean(yhat; dims = 2)
p = plot(validate.dates, validate.y; color = :black, label = "Data");
p = plot!(p, validate.dates, yh; color = :red, label = "MAP")
savefig(p, "./outputs/rnn-har-map-comparison-validate.pdf")

rmse = sqrt(mean(abs2, validate.y .- yh))
serialize("./outputs/rnn-har-map-validation-rmse.jld", rmse)

p0 = plot(validate.dates, coeffs[1, :, 1], legend = false, title = "Intercept");
p1 = plot(validate.dates, coeffs[2, :, 1], legend = false, title = "Daily");
p2 = plot(validate.dates, coeffs[3, :, 1], legend = false, title = "Weekly");
p3 = plot(validate.dates, coeffs[4, :, 1], legend = false, title = "Monthly");
p = plot(p0, p1, p2, p3; layout = 4)
savefig(p, "./outputs/rnn-har-map-time-variation-validation.pdf")

# Test

yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1); x = test.x, y = test.y)
yh = mean(yhat; dims = 2)
rmse = sqrt(mean(abs2, test.y .- yh))
serialize("./outputs/rnn-har-map-test-rmse.jld", rmse)
# ------------------------------------------------------------------------------
# Variational Inference via Bayes By Backprop
# 
# Notes: 
# ------------------------------------------------------------------------------
Random.seed!(6150533)
q, params, losses = bbb(bnn, 10, 500; mc_samples = 1, opt = Flux.ADAM())
serialize("./outputs/rnn-har-bbb-q.jld", q)
serialize("./outputs/rnn-har-bbb-params.jld", params)
serialize("./outputs/rnn-har-bbb-losses.jld", losses)
p = plot(losses[1:end]; yaxis = :log)
Random.seed!(6150533)
ch = rand(q, 10_000)

# Training Data Performance 

yhat, coeffs, lps = naive_prediction(bnn, ch)
yh = vec(mean(yhat; dims = 2))
rmse = sqrt(mean(abs2, bnn.y - yh))
serialize("./outputs/rnn-har-bbb-train-rmse.jld", rmse)

p = plot(train.dates, bnn.y; color = :black, label = "Data");
p = plot!(p, train.dates, yh; color = :red, label = "Prediction")
savefig(p, "./outputs/rnn-har-bbb-comparison.pdf")


p0 = plot(train.dates, mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(train.dates, mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(train.dates, mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(train.dates, mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-bbb-time-variation-train.pdf")

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch)
p = plot_quantile_comparison(bnn.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-bbb-qqplot-train.pdf")

# Validation data performance 
yhat, coeffs, lps = naive_prediction(bnn, ch; x = validate.x, y = validate.y)
yh = vec(mean(yhat; dims = 2))
rmse = sqrt(mean(abs2, validate.y - yh))
serialize("./outputs/rnn-har-bbb-validate-rmse.jld", rmse)

p = plot(validate.dates, validate.y; color = :black, label = "Data")
p = plot!(p, validate.dates, yh; color = :red, label = "Prediction")
savefig(p, "./outputs/rnn-har-bbb-comparison-validation.pdf")

p0 = plot(validate.dates, mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(validate.dates, mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(validate.dates, mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(validate.dates, mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-bbb-time-variation-validate.pdf")

Random.seed!(6150533)
idx_plot = sample(1:size(coeffs, 3), 10; replace = false)
p = plot(validate.dates, coeffs[2, :, idx_plot], legend = false)
savefig(p, "./outputs/rnn-har-bbb-time-variation-validate-random.pdf")

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = validate.x)
p = plot_quantile_comparison(validate.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-bbb-qqplot-validate.pdf")

# Test data performance 
yhat, coeffs, lps = naive_prediction(bnn, ch; x = test.x, y = test.y)
yh = vec(mean(yhat; dims = 2))
rmse = sqrt(mean(abs2, test.y - yh))
serialize("./outputs/rnn-har-bbb-test-rmse.jld", rmse)

p = plot(test.dates, test.y; color = :black, label = "Data")
p = plot!(p, test.dates, yh; color = :red, label = "Prediction")
savefig(p, "./outputs/rnn-har-bbb-comparison-test.pdf")

p0 = plot(test.dates, mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(test.dates, mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(test.dates, mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(test.dates, mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-bbb-time-variation-test.pdf")

Random.seed!(6150533)
idx_plot = sample(1:size(coeffs, 3), 10; replace = false)
p = plot(test.dates, coeffs[2, :, idx_plot], legend = false)
savefig(p, "./outputs/rnn-har-bbb-time-variation-test-random.pdf")

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = test.x)
p = plot_quantile_comparison(test.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-bbb-qqplot-test.pdf")

# ------------------------------------------------------------------------------
# RNN-HAR Full Bayesian
# 
# Notes: 
# ------------------------------------------------------------------------------

function cal_chain_stats(bnn, ch)
    yhat, coeffs, lps = naive_prediction(bnn, ch)
    rhat_intercept = summarystats(Chains(coeffs[1,:,:]'))[:,7]
    ess_intercept = summarystats(Chains(coeffs[1,:,:]'))[:,6]
    rhat_daily = summarystats(Chains(coeffs[2,:,:]'))[:,7]
    ess_daily = summarystats(Chains(coeffs[2,:,:]'))[:,6]
    rhat_weekly = summarystats(Chains(coeffs[3,:,:]'))[:,7]
    ess_weekly = summarystats(Chains(coeffs[3,:,:]'))[:,6]
    rhat_monthly = summarystats(Chains(coeffs[4,:,:]'))[:,7]
    ess_monthly = summarystats(Chains(coeffs[4,:,:]'))[:,6]
    return [maximum.([rhat_intercept, rhat_daily, rhat_weekly, rhat_monthly]), minimum.([ess_intercept, ess_daily, ess_weekly, ess_monthly])]
end


Random.seed!(6150533)
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 100, 500, opt; showprogress = true)

madapter = FixedMassAdapter()
sampl = SGNHTS(3f-3, 100f0; xi = 50f0^2, μ = 50f0, madapter = madapter)
ch = mcmc(bnn, 100, 100_000, sampl; θstart = copy(θmap))

serialize("./outputs/rnn-har-sgnhts-chain.jld", ch)
serialize("./outputs/rnn-har-sgnhts-sampler.jld", sampl)

ch = ch[:, 50_001:end]
chain_stats = cal_chain_stats(bnn, ch)
serialize("./outputs/rnn-har-sgnhts-chain-stats.jld", chain_stats)

yhat, coeffs, lps = naive_prediction(bnn, ch)

p0 = plot(mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-sgnhts-time-variation-train.pdf")

yh = mean(yhat; dims = 2)
p = plot(bnn.y; color = :black, label = "Data");
p = plot!(p, yh; color = :red, label = "MAP")
savefig(p, "./outputs/rnn-har-sgnhts-comparison-train.pdf")

rmse = sqrt(mean(abs2, bnn.y .- yh))
serialize("./outputs/rnn-har-sgnhts-rmse-train.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch)
p = plot_quantile_comparison(bnn.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-sgnhts-qqplot.pdf")

# Validation data

yhat, coeffs, lps = naive_prediction(bnn, ch; x = validate.x, y = validate.y)

p0 = plot(mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-sgnhts-time-variation-validation.pdf")

yh = mean(yhat; dims = 2)
p = plot(validate.dates, validate.y; color = :black, label = "Data");
p = plot!(p, validate.dates, yh; color = :red, label = "MAP")
savefig(p, "./outputs/rnn-har-sgnhts-comparison-validation.pdf")

rmse = sqrt(mean(abs2, validate.y .- yh))
serialize("./outputs/rnn-har-sgnhts-rmse-validation.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = validate.x)
p = plot_quantile_comparison(validate.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-sgnhts-qqplot-validation.pdf")

# Test data

yhat, coeffs, lps = naive_prediction(bnn, ch; x = test.x, y = test.y)

p0 = plot(mean(coeffs[1, :, :]; dims = 2), legend = false,);
p1 = plot(mean(coeffs[2, :, :]; dims = 2), legend = false,);
p2 = plot(mean(coeffs[3, :, 1]; dims = 2), legend = false);
p3 = plot(mean(coeffs[4, :, 1]; dims = 2), legend = false);
p = plot(p0, p1, p2, p3; layout = (4, 1))
savefig(p, "./outputs/rnn-har-sgnhts-time-variation-test.pdf")

yh = mean(yhat; dims = 2)
p = plot(test.dates, test.y; color = :black, label = "Data");
p = plot!(p, test.dates, yh; color = :red, label = "MAP")
savefig(p, "./outputs/rnn-har-sgnhts-comparison-test.pdf")

rmse = sqrt(mean(abs2, test.y .- yh))
serialize("./outputs/rnn-har-sgnhts-rmse-test.jld", rmse)

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch; x = test.x)
p = plot_quantile_comparison(test.y, posterior_yhat)
savefig(p, "./outputs/rnn-har-sgnhts-qqplot-test.pdf")