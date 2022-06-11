using CSV, DataFrames
using DataFramesMeta
using Dates
using Random, Distributions
using StatsPlots

"""
Load Realised Library
"""
function load_data(path = "../../data/oxfordmanrealizedvolatilityindices.csv")
    file = CSV.File(path)
    df = DataFrame(file)

    # General filtration for the data we want
    @chain df begin
        rename!(:Column1 => :date)
        @select!(:date, :Symbol, :rv5)
        @subset!(:Symbol .== ".SPX")
        @rtransform!(:rv5 = log(:rv5))
    end

    # Transforming dates
    @chain df begin
        @rtransform!(:date = DateTime(:date[1:end-7], dateformat"y-m-d H:M:S"))
    end

    # Putting it into the right order
    df = @orderby(df, :date)
    return df 
end

"""
Apply function `f` over rolling window of length `windowlength`
"""
function rollf(x::AbstractVector{T}, f::Function, windowlength::Int) where {T}
    roll = [f(x[i-windowlength+1:i]) for i=windowlength:length(x)]
    padding = fill(missing, windowlength-1)
    return vcat(padding, roll)
end

"""

Create a Tensor used for RNNs 

Given an input matrix of dimensions timesteps×features transform it into a
Tensor of dimension timesteps×features×sequences where the sequences are
overlapping subsequences of length `seq_to_one_length` of the orignal
`timesteps` long sequence
"""
function make_rnn_tensor(m::Matrix{T}, seq_to_one_length = 10) where {T}
    nfeatures = size(m, 2)
    nsequences = size(m, 1) - seq_to_one_length + 1

    tensor = Array{T}(undef, seq_to_one_length, nfeatures, nsequences)
    for i=1:nsequences
        tensor[:, :, i] = m[i:i+seq_to_one_length-1, :]
    end

    return tensor
end

"""
Create HAR Data Matrix [rv_daily, rv_weekly, rv_monthly]
"""
function get_rv_matrix(df, T = Float32)
    rv = df[!, :rv5]
    rv_daily = copy(rv)
    rv_weekly = rollf(rv_daily, mean, 5)
    rv_monthly = rollf(rv_daily, mean, 22)
    rv_matrix = hcat(rv_daily, rv_weekly, rv_monthly)
    # First 21 will be missing due to calculation of monthly rv 
    rv_matrix = rv_matrix[22:end, :]
    return T.(rv_matrix), df[!, :date][22:end]
end

"""
Get x, y, dates in RNN format.
"""
function get_x_y_dates(args...; subsequence_length = 10, T = Float32)
    df = load_data(args...)
    rv_matrix, _ = get_rv_matrix(df)
    # We add one because we want to use `subsequence_length` long 
    # subsequences to predict the next value which will be the last row 
    # in the tensor
    rv_tensor = make_rnn_tensor(rv_matrix, subsequence_length + 1)
    # We will predict one period ahead
    # and only daily rv
    y = rv_tensor[end, 1, :]
    # remove last row which is the value predicted
    x = rv_tensor[1:end-1, :, :]
    # Also returning dates for sanity
    dates = df[!, :date]
    # We lost 21 due to averaging another `sequence_length - 1` due to the subsequence embedding 
    # and another 1 due to wanting to predict one period ahead. 
    dates = dates[(22+subsequence_length-1+1):end]

    return T.(x), T.(y), dates
end

"""

Split HAR data into train, validate, test

Splits over the last dimension.

# Arguments
- `x::Array{T, 3}` is a tensor of dimension timesteps×features×sequences 
- `y::Vector{T}` is a vector containing the values to predict
- `dates` is a vector containing to the dates of y 
- `validate_from_date` a DateTime from which to validate. Stops at `test_from_date`
- `test_from_date` a DateTime from which to test.
"""
function split_train_validate_test(x::Array{T, N1}, y::Array{T, N2}, dates::Vector{DateTime}, 
    validate_from_date = DateTime("2012-01-01"), 
    test_from_date = DateTime("2015-01-01")) where {T, N1, N2}

    validate_from_index = findfirst(d -> d > validate_from_date, dates)
    test_from_index = findfirst(d -> d > test_from_date, dates)

    y_train = T.(selectdim(y, N2, 1:validate_from_index-1))
    y_validate = T.(selectdim(y, N2, validate_from_index:test_from_index-1))
    y_test = T.(selectdim(y, N2, test_from_index:size(y, N2)))

    x_train = T.(selectdim(x, N1, 1:validate_from_index-1))
    x_validate = T.(selectdim(x, N1, validate_from_index:test_from_index-1))
    x_test = T.(selectdim(x, N1, test_from_index:size(x, N1)))

    # y_train = y[1:validate_from_index-1]
    # y_validate = y[validate_from_index:test_from_index-1]
    # y_test = y[test_from_index:end]

    # Each subsequence belongs to one y. So we need to subset the subsequences.
    # x_train = x[:, :, 1:validate_from_index-1]
    # x_validate = x[:, :, validate_from_index:test_from_index-1]
    # x_test = x[:, :, test_from_index:end]

    return (x = x_train, y = y_train, dates = dates[1:validate_from_index-1]), 
        (x = x_validate, y = y_validate, dates = dates[validate_from_index:test_from_index-1]), 
        (x = x_test, y = y_test, dates = dates[test_from_index:end])
end

"""
Standardise data using statistics only taken from `train`.

# Return Values 
- `train_scales`, `validate_scaled`, `test_scaled`: train, validate, test tuples
  of the form `(x, y)` standardised using statistics from tain.
- `rescale_x`: Function taking a x-tensor as input and scaling it back. 
- `rescale_y`: Function taking a y-vector as input and scaling it back.
"""
function standardise_train_validate_test(train, validate, test)
    μy = mean(train.y)
    σy = sqrt(var(train.y))

    # Only the first row of each slice needs to be taken. Taking more data 
    # would result in duplicate observations 
    μx = mean(train.x[1,:,:]'; dims = 1)
    σx = sqrt.(var(train.x[1,:,:]'; dims = 1))

    scale_y(y) = (y .- μy)./σy
    scale_x(x) = (x .- μx)./σx
    rescale_y(y) = σy.*y .+ μy
    rescale_x(x) = σx.*x .+ μx

    train_scaled = (x = scale_x(train.x), y = scale_y(train.y))
    validate_scaled = (x = scale_x(validate.x), y = scale_y(validate.y))
    test_scaled = (x = scale_x(test.x), y = scale_y(test.y))

    return train_scaled, validate_scaled, test_scaled, rescale_x, rescale_y
end
