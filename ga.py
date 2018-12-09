import numpy as np
import os.path
import time
from operator import methodcaller, attrgetter

class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def evaluate():
        pass

class GeneticAlgorithm:
    def __init__(self, ind_gen, select, crossover, mutate):
        self.ind_gen = ind_gen
        self.select = select
        self.crossover = crossover
        self.mutate = mutate

    def run(self, **params):
        np.random.seed(params['seed'] if 'seed' in params else None)

        generation = 0

        if 'file' in params and os.path.exists(params['file']):
            info = list(map(list, np.load(params['load'])))
            
            population = info[-1]
        else:
            params['file'] = params.get('file', 
                    'runs/{}.npy'.format(time.strftime("%Y%m%d-%H%M%S")))
            info = []

            population = [self.ind_gen() for _ in range(params['N'])]

        info.append(population)
        np.save(params['file'], info)

        print('########### GENERATION {} ###########'.format(generation + 1))

        while generation < params['G']:
            for i, ind in enumerate(population):
                print('########### EVALUATING {}/{} ###########'.format(i + 1, params['N']))
                ind.evaluate(1)

            new_population = []

            elite = sorted(population, reverse = True, key = attrgetter('fitness'))

            if 'top_eval' in params:
                for i, ind in enumerate(population[:params['top_eval'][0]]):
                    print('########### EVALUATING TOP {}/{} ###########' \
                        .format(i + 1, params['top_eval'][0]))
                    ind.evaluate(params['top_eval'][1])

            elite = elite[:params['elitism']]

            if len(elite) > 0:
                new_population.extend(elite)

            while len(new_population) < params['N']:
                chosen_ind = self.select(population, 1)[0]

                new_population.append(self.mutate(chosen_ind))

            info.append(population)
            np.save(params['file'], info)

            population = new_population
            generation += 1

            print('########### GENERATION {} ###########'.format(generation + 1))

        for ind in population:
            ind.evaluate()

        info.append(population)
        np.save(params['file'], info)

        return sorted(population, reverse = True, key = attrgetter('fitness'))[0], info