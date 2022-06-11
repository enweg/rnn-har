# ------------------------------------------------------------------------------
# Simulated Time Varying Coefficient HAR model
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

function naive_prediction(bnn, draws::Array{T, 2}) where {T}
    coeffs = Array{T, 3}(undef, 4, length(bnn.y), size(draws,2))
    yhats = Array{T, 2}(undef, length(bnn.y), size(draws, 2))
    lps = Array{T, 1}(undef, size(draws, 2))
    Threads.@threads for i=1:size(draws, 2)
        lps[i] = loglikeprior(bnn, draws[:,i], bnn.x, bnn.y)
        net = bnn.like.nc(draws[:, i])
        c = [net(xx) for xx in eachslice(bnn.x[1:end-1,1:3,:]; dims = 1)][end]
        xhar = bnn.x[end, :, :]
        yh = c[1,:] .+ c[2,:].*xhar[1,:] .+ c[3,:].*xhar[2,:] .+ c[4,:].*xhar[3,:]
        coeffs[:,:,i] = c
        yhats[:,i] = yh
    end
    return yhats, coeffs, lps
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
# ------------------------------------------------------------------------------

df = load_data()
rv_matrix, dates = get_rv_matrix(df)
# We want to predict one day ahead
x = permutedims(rv_matrix[1:end-1,:], [2, 1])
y = rv_matrix[2:end, 1]
train, validate, test = split_train_validate_test(x, y, dates[2:end])

# ------------------------------------------------------------------------------
# Data Simulation
# 
# Notes: 
# ------------------------------------------------------------------------------

function simulate(n, rng)
    burn = 100
    N = n + burn + 22
    y = zeros(N)
    y[1:22] = randn(rng, 22)
    βs = zeros(3, N)
    s1 = [0.3, 0.2, 0.4]
    s2 = [0.8, 0.1, 0.05]
    s3 = [0.05, 0.6, 0.2]
    f1(x) = sigmoid(19*x + 9)
    f2(x) = sigmoid(9*x - 3)
    for t=24:N
        # monthly = mean(y[t-23:t-2])
        monthly = y[t-2]
        z1 = f1(monthly)
        z2 = f2(monthly)
        βs[:,t] = (1-z1)*s1 .+ z1*(z2*s2 + (1-z2)*s3)

        monthly = mean(y[t-22:t-1])
        weekly = mean(y[t-5:t-1])
        y[t] = βs[1, t]*y[t-1] + βs[2,t]*weekly + βs[3,t]*monthly + 1*randn(rng)
    end
    return y[end-n+1:end], βs[:, end-n+1:end]
end


rng = Random.MersenneTwister(6150533)
y, beta = simulate(length(train.y), rng)
serialize("./outputs/simulations-data-y.jld", y)
serialize("./outputs/simulations-data-beta.jld", beta)
py = plot(y, label = "y");
pautocor = plot(autocor(y), label = "ACF");
p = plot(py, pautocor; layout = (2, 1))
savefig(p, "./outputs/simulations-y-and-acf.pdf")
p = plot(beta'; layout = (3, 1))
savefig(p, "./outputs/simulations-beta.pdf")

# ------------------------------------------------------------------------------
# OLS
# 
# Notes: 
# ------------------------------------------------------------------------------

xdaily = y
xweekly = rollf(y, mean, 5)
xmonthly = rollf(y, mean, 22)
datamat = [xdaily xweekly xmonthly]
datamat = Float32.(datamat[22:end, :])

x = permutedims(datamat[1:end-1, :], [2, 1])
y = datamat[2:end, 1]

βols = inv(x*x')*x*y
yhat = x'*βols
p = plot(y; color = :black, label = "Simulated Data");
p = plot!(p, yhat; color = :red, label = "OLS")
savefig(p, "./outputs/simulations-old-comparison.pdf")

rmse = sqrt(mean(abs2, y .- yhat))
serialize("./outputs/simulations-ols-rmse.jld", rmse)

# ------------------------------------------------------------------------------
# RNN-HAR MAP
# 
# Notes: 
# ------------------------------------------------------------------------------
Random.seed!(6150533)
rnntensor = make_rnn_tensor(datamat, 22)
xrnn = rnntensor[1:end-1, :, :]
yrnn = rnntensor[end, 1, :]

net = Flux.Chain(RNN(3, 3), Dense(3, 4))
nc = destruct(net)
prior = GaussianPrior(nc, 0.5f0)
like = HARLikelihood(nc, Gamma(2.0, 0.5))
init = InitialiseAllSame(Normal(0f0, 0.1f0), like, prior)
bnn = BNN(xrnn, yrnn, like, prior, init)

# MAP Estimate 

Random.seed!(6150533)
opt = FluxModeFinder(bnn, Flux.ADAM())
θmap = find_mode(bnn, 100, 500, opt)
serialize("./outputs/simulations-map-theta.jld", θmap)

yhat, coeffs, lps = naive_prediction(bnn, reshape(θmap, :, 1))
yh = mean(yhat; dims = 2)
p = plot(bnn.y; color = :black, label = "Data");
p = plot!(p, yh; color = :red, label = "MAP")
savefig(p, "./outputs/simulations-map-comparison.pdf")

rmse = sqrt(mean(abs2, bnn.y .- yh))
serialize("./outputs/simulations-map-rmse.jld", rmse)

p1 = plot(coeffs[2, :, 1], legend = false, title = "Daily");
p1 = plot!(p1, beta[1,end-size(coeffs,2)+1:end]);
p2 = plot(coeffs[3, :, 1], legend = false, title = "Weekly");
p2 = plot!(p2, beta[2,end-size(coeffs,2)+1:end]);
p3 = plot(coeffs[4, :, 1], legend = false, title = "Monthly");
p3 = plot!(p3, beta[3,end-size(coeffs,2)+1:end]);
p = plot(p1, p2, p3; layout = (3, 1))
savefig(p, "./outputs/simulations-map-time-variation.pdf")

# ------------------------------------------------------------------------------
# Variational Inference
# 
# Notes: 
# ------------------------------------------------------------------------------

Random.seed!(6150533)
q, params, losses = bbb(bnn, 10, 500; mc_samples = 1, opt = Flux.ADAM())
serialize("./outputs/simulations-bbb-q.jld", q)
serialize("./outputs/simulations-bbb-params.jld", params)
serialize("./outputs/simulations-bbb-losses.jld", losses)
p = plot(losses[1:end]; yaxis = :log)

Random.seed!(6150533)
ch = rand(q, 10_000)

yhat, coeffs, lps = naive_prediction(bnn, ch)
yh = vec(mean(yhat; dims = 2))
rmse = sqrt(mean(abs2, bnn.y - yh))
serialize("./outputs/simulations-bbb-rmse.jld", rmse)

p = plot(bnn.y; color = :black, label = "Data");
p = plot!(p, yh; color = :red, label = "Prediction")
savefig(p, "./outputs/simulations-bbb-comparison.pdf")

p1 = plot(mean(coeffs[2, :, :]; dims = 2), legend = false,);
p1 = plot!(p1, beta[1,end-size(coeffs,2)+1:end]);
p2 = plot(mean(coeffs[3, :, 1]; dims = 2), legend = false);
p2 = plot!(p2, beta[2,end-size(coeffs,2)+1:end]);
p3 = plot(mean(coeffs[4, :, 1]; dims = 2), legend = false);
p3 = plot!(p3, beta[3,end-size(coeffs,2)+1:end]);
p = plot(p1, p2, p3; layout = (3, 1))
savefig(p, "./outputs/simulations-bbb-time-variation.pdf")

posterior_yhat = sample_posterior_predict(bnn, ch)
p = plot_quantile_comparison(bnn.y, posterior_yhat)
savefig(p, "./outputs/simulations-bbb-qqplot.pdf")

# ------------------------------------------------------------------------------
# RNN-HAR Full Bayesian
# 
# Notes: 
# ------------------------------------------------------------------------------

# SGNHT -
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
sampl = SGNHTS(8f-3, 5f0; xi = 5f0^2, μ = 5f0, madapter = madapter)
ch = mcmc(bnn, 100, 100_000, sampl; θstart = copy(θmap))

serialize("./outputs/simulations-sgnhts-sampler.jld", sampl)
serialize("./outputs/simulations-sgnhts-chain.jld", ch)

ch = ch[:, 50_001:end]
chain_stats = cal_chain_stats(bnn, ch)
serialize("./outputs/simulations-sgnhts-chain-stats.jld", chain_stats)

yhat, coeffs, lps = naive_prediction(bnn, ch)
yh = vec(mean(yhat; dims = 2))
rmse = sqrt(mean(abs2, bnn.y - yh))
serialize("./outputs/simulations-sgnhts-rmse.jld", rmse)

p = plot(bnn.y; color = :black, label = "Data");
p = plot!(p, yh; color = :red, label = "Prediction")
savefig(p, "./outputs/simulations-sgnhts-comparison.pdf")

p1 = plot(mean(coeffs[2, :, :]; dims = 2), legend = false,);
p1 = plot!(p1, beta[1,end-size(coeffs,2)+1:end]);
p2 = plot(mean(coeffs[3, :, 1]; dims = 2), legend = false);
p2 = plot!(p2, beta[2,end-size(coeffs,2)+1:end]);
p3 = plot(mean(coeffs[4, :, 1]; dims = 2), legend = false);
p3 = plot!(p3, beta[3,end-size(coeffs,2)+1:end]);
p = plot(p1, p2, p3; layout = (3, 1))
savefig(p, "./outputs/simulations-sgnhts-time-variation.pdf")

Random.seed!(6150533)
posterior_yhat = sample_posterior_predict(bnn, ch)
p = plot_quantile_comparison(bnn.y, posterior_yhat)
savefig(p, "./outputs/simulations-sgnhts-qqplot.pdf")