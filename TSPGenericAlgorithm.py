import random
import tsplib95
import numpy as np
import matplotlib.pyplot as plt

def plot_tour(problem, tour):
    coordinates = [problem.get_display(tour[i]) for i in range(len(tour))]

    x, y = zip(*coordinates)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', mfc='r') # Plot the cities and the path
    plt.title("Best Tour")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")

    for i in range(len(tour)):
        plt.annotate(tour[i], (x[i], y[i]))

    plt.show()

def create_initial_population(problem, population_size):
    nodes = list(problem.get_nodes())  
    nodes.remove(1)  

    population = []
    for _ in range(population_size):
        tour = random.sample(nodes, len(nodes))
        tour = [1] + tour + [1]
        population.append(tour)

    return population

def calculate_distance(problem, tour):
    total_distance = 0   
   
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i+1]
        
        distance = problem.get_weight(current_city, next_city)
        
        total_distance += distance
    
    return total_distance 


def fitness(problem, tour):
    return 1 / calculate_distance(problem, tour)

def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    parent_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
    selected_population = []
    for i in parent_indices:
        selected_population.append(population[i])
    return selected_population

def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 2)
    child = []

    for i in range(cut):
        child.append(parent1[i])

    for city in parent2:
        if city not in child:
            child.append(city)
    child.append(1)
    return child

def mutation(tour, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(tour) - 1), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

def find_best(problem, population):
    best = calculate_distance(problem, population[0])
    tourNr = 0
    for n in range(1, len(population)):
        nextBest = calculate_distance(problem, population[n])
        if nextBest < best:
                tourNr = n
        
    return population[tourNr]

def genetic_algorithm(problem, population_size, generations, mutation_rate):
    population = create_initial_population(problem, population_size)
    best_distances = []  # List to track the best distance in each generation

    for _ in range(generations):
        fitnesses = []
        for tour in population:
            fitness_value = fitness(problem, tour)
            fitnesses.append(fitness_value)

        bestFromPrevious = find_best(problem, population) # elitism, saves the best tour after each iteration so the evolution continues to be have better paths

        new_population = [bestFromPrevious]
        for _ in range(population_size):
            parent1, parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population

        current_best_distance = min(calculate_distance(problem, tour) for tour in population)
        best_distances.append(current_best_distance)

        # print a simple progress bar
        
    best_tour = max(population, key=lambda tour: fitness(problem, tour))
    return best_tour, best_distances

def plot_evolution(best_distances):
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances, marker='o')
    plt.title("Evolution of the Minimum Total Traveling Distance")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid()
    plt.show()

file_path = 'datasets/dataset14.tsp'
problem = tsplib95.load(file_path)

# population = how many solutions will be in a generation
# generations = number of iterations
# mutation_rate = probability of mutation, should experiment with different datasets

best_solution, best_distances = genetic_algorithm(problem, population_size=50, generations=100, mutation_rate=0.2)
best_solution_distance = calculate_distance(problem, best_solution)

print("Best solution:", best_solution)
print("Best solution distance:", best_solution_distance)
plot_evolution(best_distances)
plot_tour(problem, best_solution)
