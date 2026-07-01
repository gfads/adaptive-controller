import numpy as np
from ..models.pid_model import PIDModel

class PSOTuner:
    """
    Simple Particle Swarm Optimization (PSO) tuner for KP/KI/KD.
    Lightweight implementation intended for batch optimization on last N samples.
    """

    def __init__(
        self,
        n_particles=30,
        iterations=60,
        w=0.7,        # inertia
        c1=1.4,       # cognitive
        c2=1.4,       # social
        kp_range=(1e-6, 0.01),
        ki_range=(1e-6, 0.01),
        kd_range=(0.0, 0.01),
    ):
        self.n_particles = int(n_particles)
        self.iterations = int(iterations)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range

    def _clip(self, vec):
        kp = np.clip(vec[0], *self.kp_range)
        ki = np.clip(vec[1], *self.ki_range)
        kd = np.clip(vec[2], *self.kd_range)
        return np.array([kp, ki, kd], dtype=float)

    def tune(self, df):
        pid = PIDModel()

        # initialize particles uniformly in ranges
        particles = np.vstack([
            np.random.uniform(self.kp_range[0], self.kp_range[1], self.n_particles),
            np.random.uniform(self.ki_range[0], self.ki_range[1], self.n_particles),
            np.random.uniform(self.kd_range[0], self.kd_range[1], self.n_particles),
        ]).T  # shape (n_particles, 3)

        velocities = np.zeros_like(particles)

        # personal bests
        pbest = particles.copy()
        pbest_scores = np.full(self.n_particles, np.inf)
        # global best
        gbest = None
        gbest_score = np.inf

        # evaluate initial
        for i in range(self.n_particles):
            pid.update(*particles[i])
            score = pid.evaluate_batch(df)
            pbest_scores[i] = score
            if score < gbest_score:
                gbest_score = score
                gbest = particles[i].copy()

        # PSO loop
        for t in range(self.iterations):
            r1 = np.random.rand(self.n_particles, 3)
            r2 = np.random.rand(self.n_particles, 3)

            # velocity update
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - particles)
                + self.c2 * r2 * (gbest - particles)
            )

            # position update
            particles = particles + velocities
            # clip to ranges
            particles = np.vstack([self._clip(p) for p in particles])

            # evaluate
            for i in range(self.n_particles):
                pid.update(*particles[i])
                score = pid.evaluate_batch(df)
                # update pbest
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()
                # update gbest
                if score < gbest_score:
                    gbest_score = score
                    gbest = particles[i].copy()
    
        return {
                "kp": float(gbest[0]),
                "ki": float(gbest[1]),
                "kd": float(gbest[2])
            }

    
    
