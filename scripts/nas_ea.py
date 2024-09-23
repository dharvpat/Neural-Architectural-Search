import random
import torch

# Mutate a given architecture by randomly changing a layer configuration
def mutate_architecture(architecture):
    mutated_architecture = architecture.copy()
    layer_to_mutate = random.choice(range(len(architecture)))

    # Randomly mutate one of the layer parameters
    mutated_architecture[layer_to_mutate] = (
        architecture[layer_to_mutate][0],  # Keep the number of output channels
        random.choice([3, 5, 7]),          # Change kernel size
        random.choice([1, 2])              # Change stride
    )

    return mutated_architecture

# Crossover between two architectures
def crossover_architectures(parent1, parent2):
    split_point = random.randint(0, len(parent1) - 1)
    return parent1[:split_point] + parent2[split_point:]

# Evolutionary Algorithm loop
def evolutionary_search(population, search_space, trainloader, testloader, num_generations=50):
    for generation in range(num_generations):
        population_performance = []
        for architecture in population:
            model = search_space.generate(architecture)
            reward = get_reward(model, trainloader, testloader, device='cuda')
            population_performance.append((architecture, reward))

        # Sort population by reward
        population_performance.sort(key=lambda x: x[1], reverse=True)

        # Keep top half, and mutate/crossover the rest
        new_population = [arch for arch, _ in population_performance[:len(population) // 2]]
        while len(new_population) < len(population):
            if random.random() > 0.5:
                # Mutation
                parent = random.choice(new_population)
                new_population.append(mutate_architecture(parent))
            else:
                # Crossover
                parent1, parent2 = random.sample(new_population, 2)
                new_population.append(crossover_architectures(parent1, parent2))

        population = new_population

    return population_performance[0]  # Return the best architecture