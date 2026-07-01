import numpy as np
from ..models.pid_model import PIDModel


class EvolutionTuner:
    """
    Algoritmo evolutivo simples para otimizar KP/KI/KD.
    """

    def __init__(
        self,
        population_size=40,
        generations=50,
        mutation_rate=0.2,
        kp_range=(1e-6, 0.01),
        ki_range=(1e-6, 0.01),
        kd_range=(0.0, 0.01),
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range

    # ----------------------------------------------------------------------
    def random_individual(self):
        return np.array([
            np.random.uniform(*self.kp_range),
            np.random.uniform(*self.ki_range),
            np.random.uniform(*self.kd_range),
        ])

    # ----------------------------------------------------------------------
    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(3)
            ranges = [self.kp_range, self.ki_range, self.kd_range]
            individual[idx] = np.random.uniform(*ranges[idx])
        return individual

    # ----------------------------------------------------------------------
    def tune(self, df):

        pid = PIDModel()

        # inicial population
        pop = np.array([self.random_individual() for _ in range(self.population_size)])

        for _ in range(self.generations):

            # evaluate fitness
            fitness = []
            for ind in pop:
                pid.update(*ind)
                fitness.append(pid.evaluate_batch(df))

            fitness = np.array(fitness)

            # select top 50%
            sorted_idx = np.argsort(fitness)
            survivors = pop[sorted_idx[: len(pop) // 2]]

            # reproduce
            children = []
            for _ in range(len(pop) // 2):
                p1, p2 = survivors[np.random.randint(len(survivors), size=2)]
                child = (p1 + p2) / 2
                child = self.mutate(child)
                children.append(child)

            pop = np.vstack([survivors, children])

        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        return {
            "kp": float(best[0]),
            "ki": float(best[1]),
            "kd": float(best[2])
        }
