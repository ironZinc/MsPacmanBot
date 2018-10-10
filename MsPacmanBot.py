import gym
import numpy as np
import time
import random
from statistics import mean, median
from collections import Counter

env = gym.make("Pong-ram-v0").env

hl1_num_nodes = 15
output_length = 6

initial_pop_size = 60
num_training_generations = 100
#breeding_transfer_error_rate = 0.2
breeding_best_sample_size = 6
breeding_lucky_few_size = 0
num_child_per_couple = 20
#mutation_rate = 0.2
breeding_transfer_error_rate = 0.0007
mutation_rate = 0.001


class Individual:
    def __init__(self, dna=[], empty=False):
        if empty:
            self.genome = []
        elif dna == []:
            self.fitness = 0
            self.hl1_w = np.random.uniform(-1, 1, [128, hl1_num_nodes])
            self.hl1_b = np.random.uniform(-1, 1, [1, hl1_num_nodes])
            self.output_w = np.random.uniform(-1, 1, [hl1_num_nodes, output_length])
            self.output_b = np.random.uniform(-1, 1, [1, output_length])
            self.genome = np.concatenate((np.reshape(self.hl1_w, [-1]), np.reshape(self.hl1_b, [-1]), np.reshape(self.output_w, [-1]), np.reshape(self.output_b, [-1])))
        else:
            self.fitness = 0
            self.genome = dna
            hl1_1d_size_w = 128 * hl1_num_nodes
            hl1_1d_size = hl1_1d_size_w + hl1_num_nodes
            output_1d_size_w = hl1_1d_size + hl1_num_nodes * output_length
            output_1d_size = output_1d_size_w + output_length
            self.hl1_w = np.reshape(dna[0:hl1_1d_size_w], [128, hl1_num_nodes])
            self.hl1_b = np.reshape(dna[hl1_1d_size_w:hl1_1d_size], [1, hl1_num_nodes])
            self.output_w = np.reshape(dna[hl1_1d_size:output_1d_size_w], [hl1_num_nodes, output_length])
            self.output_b = np.reshape(dna[output_1d_size_w:output_1d_size], [1, output_length])

    def sigmoid(self,x):
        return 1/(1+np.exp(np.negative(x)))

    def compute_next_move(self, input_val):
        input_val = np.array(input_val).reshape([1, 128])
        #print(input_val)
        hl1_output = self.sigmoid(np.dot(input_val, self.hl1_w) + self.hl1_b)
        #print(hl1_output)
        final_output = self.sigmoid(np.dot(hl1_output, self.output_w) + self.output_b)
        #print(final_output)
        #print(str(len(final_output)))
        #print(str(np.argmax(final_output)))
        return np.argmax(final_output)

    def play_games(self, num_games, set_self_fitness=True, see_games=False):
        for game in range(num_games):
            env.reset()
            previous_observation = []
            while True:
                if see_games:
                    env.render()
                    time.sleep(0.05)
                if previous_observation == []:
                    observation, reward, done, info = env.step(np.random.randint(0, output_length))
                else:
                    observation, reward, done, info = env.step(self.compute_next_move(previous_observation))
                previous_observation = observation
                if set_self_fitness:
                    self.fitness += reward
                if done:
                    previous_observation = []
                    break
        if set_self_fitness:
            return self.fitness

    def mutate(self, mutation_rate):
        #mutation rate is in percent
        for i in range(len(self.genome)):
            if np.random.uniform(0, 1) <= mutation_rate:
                self.genome[i] = np.random.uniform(-1, 1)


def breed(ind1, ind2):
    child_genome = []
    for i in range(len(ind1.genome)):
        if np.random.randint(0, 2) == 0:
            child_genome.append(ind1.genome[i])
            if np.random.uniform(0, 1) <= breeding_transfer_error_rate:
                child_genome[i] += np.random.uniform(-0.01, 0.01) * child_genome[i]
        else:
            child_genome.append(ind2.genome[i])
            if np.random.uniform(0, 1) <= breeding_transfer_error_rate:
                child_genome[i] += np.random.uniform(-0.01, 0.01) * child_genome[i]
    return Individual(dna=child_genome)


def get_next_generation(pop):
    sorted_pop = sorted(pop, key=lambda individual: individual.fitness, reverse=True)
    max_fitness = sorted_pop[0].fitness
    print("Max fitness: " + str(max_fitness))
    breeders = []
    for i in range(breeding_best_sample_size):
        breeders.append(sorted_pop[i])
    for i in range(breeding_lucky_few_size):
        ran = np.random.randint(0, len(pop))
        breeders.append(pop[ran])
    random.shuffle(breeders)
    next_generation = []
    for i in range(int(len(breeders)/2)):
        for _ in range(num_child_per_couple):
            next_generation.append(breed(breeders[i], breeders[len(breeders) - i - 1]))
    for i in range(len(next_generation)):
        next_generation[i].mutate(mutation_rate=mutation_rate)
    return next_generation, max_fitness

# ind = Individual()
# ind.play_games(1,see_games=True)
# print(ind.fitness)

# for i in range(300):
#     observation, reward, done, info = env.step(np.random.randint(0, output_length))
#     env.render()
#     time.sleep(0.025)
#     if done:
#         break

#
previous_pop = []
current_pop = []
for i in range(initial_pop_size):
    previous_pop.append(Individual())

for i in range(num_training_generations):
    print("\nGeneration " + str(i) + " out of " + str(num_training_generations) + " generations")
    for j in range(len(previous_pop)):
        previous_pop[j].play_games(1)
    # if i % 5 == 0:
    total_fitness = 0
    for ind in previous_pop:
        total_fitness += ind.fitness
    average_fitness = total_fitness/len(previous_pop)
    print("Average fitness: " + str(average_fitness))
    current_pop, max_fitness = get_next_generation(previous_pop)
    previous_pop = current_pop
    if max_fitness >= 3000:
        break

sorted_current_pop = sorted(current_pop, key=lambda individual: individual.fitness, reverse=True)
fitness = sorted_current_pop[0].play_games(3, see_games=True, set_self_fitness=True)
good_genome = sorted_current_pop[0].genome
print(fitness)
print(good_genome)
