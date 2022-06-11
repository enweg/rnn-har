
# EBC4257 Machine Learning 

- This repo contains the code and final paper for EBC4257 @ SBE Maastricht
- Data could not be uploaded by can be obtained from the [Realized Volatility Library](https://realized.oxford-man.ox.ac.uk/data)
- The code relies on BFlux a Bayesian Neural Network library which is currently being developed for my RA and Master thesis. Since BFlux is actively changing, the paper uses a separate branch of BFlux called RNN-HAR. 

## Replicating 

1. Download data
2. Put data file in data folder and name file oxfordmanrealizedvolatilityindices.csv
3. open julia in the code/julia folder: to be able to run in parallel juliat must be opened with multiple threads activates: `julia -t 4` will start julia with 4 threads.
4. type in `using Pkg; Pkg.activate("."); Pkg.instantiate();`
5. run `include("har.jl")` to replicate the baseline results
6. run `include("rnn_har.jl")` to replicate the RNN-HAR results
7. run `include("simulated-tvc.jl")` to replicate the simulation study

If any problems come up, please open an issue or contact me via email. 

 
