# Import all key functions and classes from relevant modules
from .nas_rl import NASController, train_controller, get_reward
from .nas_ea import mutate_architecture, crossover_architectures, evolutionary_search
from .data_utils import load_cifar10
from .train_model import train_model
from .evaluate_model import evaluate_model

__all__ = [
    'NASController', 
    'train_controller', 
    'get_reward', 
    'mutate_architecture', 
    'crossover_architectures', 
    'evolutionary_search', 
    'load_cifar10', 
    'train_model', 
    'evaluate_model'
]