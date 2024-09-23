import argparse

def get_config():
    """
    Returns a configuration dictionary or argument parser for the NAS project.
    You can extend this function to include more parameters for various components
    like RL, EA, model architecture, and training settings.
    """
    parser = argparse.ArgumentParser(description='Neural Architecture Search Configuration')

    # General Configurations
    parser.add_argument('--method', type=str, default='rl', choices=['rl', 'ea'],
                        help="Choose the NAS method: 'rl' for reinforcement learning, 'ea' for evolutionary algorithm.")
    
    # RL Controller Configurations
    parser.add_argument('--rl_hidden_dim', type=int, default=64, help='Hidden size for the LSTM controller in RL.')
    parser.add_argument('--rl_input_dim', type=int, default=10, help='Input dimension for the RL controller.')
    parser.add_argument('--rl_num_layers', type=int, default=1, help='Number of layers in the RL controller LSTM.')
    parser.add_argument('--rl_epochs', type=int, default=10, help='Number of epochs to train the RL controller.')

    # Evolutionary Algorithm Configurations
    parser.add_argument('--ea_population_size', type=int, default=20, help='Population size for the evolutionary algorithm.')
    parser.add_argument('--ea_num_generations', type=int, default=10, help='Number of generations for the EA.')
    
    # Architecture and Model Configurations
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in the architectures.')
    parser.add_argument('--min_channels', type=int, default=16, help='Minimum number of channels for the layers.')
    parser.add_argument('--max_channels', type=int, default=128, help='Maximum number of channels for the layers.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the convolutional layers.')

    # Training Configurations
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train architectures.')

    # Reward Configuration
    parser.add_argument('--complexity_penalty', type=float, default=0.01, help='Penalty factor for model complexity in reward calculation.')

    args = parser.parse_args()
    return vars(args)  # Return the parsed arguments as a dictionary