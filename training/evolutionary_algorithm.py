import random
import torch
from models.search_space import create_model
from training.train import train_and_evaluate

# Mutation: Mutate an architecture by changing one of its layers
def mutate_architecture(architecture):
    mutated_architecture = architecture.copy()
    layer_to_mutate = random.randint(0, len(architecture) - 1)
    mutated_architecture[layer_to_mutate] = (
        random.randint(16, 64),  # Random change in layer width
        random.randint(3, 7),    # Random kernel size
        random.randint(1, 2)     # Random stride
    )
    return mutated_architecture

# Crossover: Combine two architectures
def crossover_architecture(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Evolutionary Algorithm
def evolutionary_search(population_size, num_generations):
    # Initialize population with random architectures
    population = [
        [(3, 32, 32, 3), (3, 64, 64, 3), (3, 128, 128, 3)] for _ in range(population_size)
    ]

    for generation in range(num_generations):
        performance = []
        # Evaluate all architectures in the population
        for architecture in population:
            reward = train_and_evaluate(architecture)
            performance.append((architecture, reward))

        # Select the top-performing architectures
        performance.sort(key=lambda x: x[1], reverse=True)
        top_architectures = [x[0] for x in performance[:population_size // 2]]

        # Create new population via mutation and crossover
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(top_architectures, 2)
            child = crossover_architecture(parent1, parent2)
            new_population.append(mutate_architecture(child))
        
        # Combine top-performing architectures with new offspring
        population = top_architectures + new_population

        print(f"Generation {generation+1} Best reward: {performance[0][1]}")
    
    return population[0]  # Return the best architecture found