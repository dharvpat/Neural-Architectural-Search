from training.controller_rl import RLController
from training.evolutionary_algorithm import evolutionary_search
from training.train import train_and_evaluate
from training.controller_rl import RLController
from training.evolutionary_algorithm import evolutionary_search
from training.train import train_and_evaluate
from utils.config import get_config

def run_rl(config):
    input_dim = config['rl_input_dim']
    hidden_dim = config['rl_hidden_dim']
    num_layers = config['num_layers']
    controller = RLController(input_dim, hidden_dim, config['rl_num_layers'])

    for _ in range(config['rl_epochs']):
        architecture = []
        for _ in range(num_layers):
            # Sample an operation and make sure the values are converted to integers
            operation, _ = controller.sample(input_dim)
            
            in_channels = int(operation) + config['min_channels']
            out_channels = max(int(operation) + config['min_channels'] + 1, 1)  # Ensure out_channels >= 1
            kernel_size = int(operation) % 3 + 3  # Example kernel size range [3, 4, 5]
            stride = 1  # Example fixed stride
            
            architecture.append((in_channels, out_channels, kernel_size, stride))

        # Train and evaluate the model with the architecture
        print(f"Proposed Architecture: {architecture}")
        reward = train_and_evaluate(architecture)
        print(f"Evaluated Architecture: {architecture} Reward: {reward}")


def run_ea(config):
    population_size = config['ea_population_size']
    num_generations = config['ea_num_generations']
    best_architecture = evolutionary_search(population_size, num_generations)
    print(f"Best Architecture found: {best_architecture}")

if __name__ == "__main__":
    config = get_config()
    method = config['method'].lower()
    if method == 'rl':
        run_rl(config)
    elif method == 'ea':
        run_ea(config)
    else:
        print("Invalid method! Please choose 'RL' or 'EA'.")