Execution instructions:

RE-INFORCEMENT LEARNING:
`python3 main.py --method rl --rl_input_dim 10 --rl_num_layers 1 --rl_epochs 10 --rl_hidden_dim 128`

EVOLUTIONARY ALGORITHM:
`python3 main.py --method ea --ea_population_size 30 --ea_num_generations 10`

Other options described in `/utils/config.py`

For the time being, only CUDA and CPU devices are suported for both methods. This may change in the future/reach out if you would like to partner on this.

Also, to understand the approach behind this project, please see /docs/approach.md`