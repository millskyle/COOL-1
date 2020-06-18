The main script is `train_val.py`.  Try running with `python train_val.py -h` to see available options.

# Examples

## Training:

`submit.s` is an example training script.  This performs validation periodically and includes support for logging to Weights and Biases. 


## Testing:
`test_all.s` is an example testing script that will test all sizes of lattice.





## Note: 
You should avoid initiating multiple runs from the same working directory at the same time if the runs have different lattice sizes or connectivity.  The SA environment creates a `latfile` file in the working directory to construct the graph (upon import). After import, it is assigned a uniquely-named temporary file in /tmp, but if multiple jobs are initialized at the same time, a simulation could use a `latfile` intended for a different computation.  Once a simulation has been running for a few seconds, it is safe to launch a different graph.




