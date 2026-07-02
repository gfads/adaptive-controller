## Algorithmic Reproducibility & Hyperparameter Specifications

To ensure the complete reproducibility of the optimization strategies evaluated in the AO-Controller framework, this section outlines the underlying mathematical formulations, search spaces, canonical hyperparameters, and academic references used across all six operational paradigms.

### 1. Cost Function (Objective Function)
All optimization algorithms (SPSA, GA, PSO, Bayes, Optuna, and RL) operate on an identical batch window of $N = 300$ error samples (defined as #Errors in the experimental setup). The common objective is to minimize system chatter and controller aggressiveness by minimizing the cumulative absolute magnitude of the computed controller output signal $u_t$:

$$J = \sum_{t=1}^{N} |u_t| = \sum_{t=1}^{N} |K_p e_t + K_i e_{i,t} + K_d e_{d,t}|$$

---

### 2. Algorithmic Configurations & Reference Matrices

#### 2.1 SPSA (Simultaneous Perturbation Stochastic Approximation)
* **Operational Context:** Gradient approximation descent scheme optimized for noisy profiles.
* **Convergence Criteria:** Runs for an invariant fixed execution budget of exactly 60 iterations per invocation.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Initial Parameter Guess** | Range Mid-point | Starts optimization from the precise center arithmetic mean of the limits |
| **Optimization Iterations** | 60 loops | Fixed budget convergence baseline |
| **Base Step Coefficient (a)**| $0.01$ | Initial numerator gain weight scaling the gradient descent step [14] |
| **Base Perturbation (c)** | $0.005$ | Initial numerator scaling factor for search direction magnitude [14] |
| **Step Decay Schedule** | $a_k = \frac{a}{k}$ | Harmonic decay reducing parameter step adjustments over time [14] |
| **Perturbation Decay** | $c_k = \frac{c}{\sqrt{k}}$ | Square-root decay scaling down exploration steps for stabilization [14] |
| **Perturbation Vector** | Rademacher Dist. | Symmetric Bernoulli trial assignment: values in $\{-1, 1\}^3$ [14] |
| **Search Bounds (Kp, Ki)** | $[0.000001, 0.01]$ | Operational limits mapped via continuous uniform clipping |
| **Search Bounds (Kd)** | $[0.0, 0.01]$ | Operational limits mapped via continuous uniform clipping |
| **Total Cost Budget** | 120 evaluations | Enforced as: 2 * iterations per calculation cycle [14] |

#### 2.2 Evolution (Genetic Algorithm)
* **Operational Context:** Global search method focused on non-convex terrain bypassing gradient computations entirely.
* **Convergence Criteria:** Enforces a rigid execution envelope terminating precisely at 50 generations.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Population Size** | 40 individuals | Size of the active candidate pool per generation [15] |
| **Generations** | 50 loops | Complete evolutionary lifecycle per runtime cycle [15] |
| **Selection Mechanism** | Truncation (Top 50%) | Truncation selection ensuring the top 20 elites survive untouched [15] |
| **Crossover Strategy** | Intermediate | Pairwise midpoint recombination blending: $\text{child} = \frac{p_1 + p_2}{2}$ [15] |
| **Mutation Rate** | 0.20 (20% chance) | Probability an offspring gene undergoes a random parameter reset [15] |
| **Search Bounds (Kp, Ki)** | $[0.000001, 0.01]$ | Bound-compliant continuous float initialization limits |
| **Search Bounds (Kd)** | $[0.0, 0.01]$ | Bound-compliant continuous float initialization limits |
| **Total Cost Budget** | 2,000 evaluations | Calculated as: Population * Generations per tuning block |

#### 2.3 PSO (Particle Swarm Optimization)
* **Operational Context:** Swarm intelligence approach balancing exploration and exploitation over non-convex landscapes.
* **Convergence Criteria:** The swarm runs for an invariant computational envelope of exactly 60 iterations.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Swarm Size** | 30 particles | Number of candidate vectors simultaneously exploring the space [16] |
| **Optimization Iterations** | 60 cycles | Total velocity/position update cycles per execution loop |
| **Inertia Weight (w)** | 0.70 | Canonical momentum scaling factor for trajectory stabilization [16] |
| **Cognitive Coeff. (c1)** | 1.40 | Pull factor toward personal best ($p_{\text{best}}$) [16] |
| **Social Coeff. (c2)** | 1.40 | Pull factor toward global swarm best ($g_{\text{best}}$) [16] |
| **Position Update Scheme** | Step Displacement | $x_{t+1} = x_t + v_{t+1}$ with automatic internal bound clipping [16] |
| **Search Bounds (Kp, Ki)** | $[0.000001, 0.01]$ | Particle coordination boundaries |
| **Search Bounds (Kd)** | $[0.0, 0.01]$ | Particle coordination boundaries |
| **Total Cost Budget** | 1,830 evaluations | Enforced as: Particles + (Particles * Iterations) per cycle [16] |

#### 2.4 Bayes (Manual Bayesian Optimization)
* **Operational Context:** Sequential Model-Based Optimization (SMBO) tracking black-box cost variance.
* **Convergence Criteria:** Terminates exactly after a fixed budget of 50 evaluations.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Surrogate Model** | Gaussian Process | Scikit-Learn GaussianProcessRegressor with normalize_y=True [17] |
| **Kernel Function** | $\text{Constant} \times \text{Matérn} + \text{WhiteNoise}$ | ConstantKernel(1.0, (1e-3, 10)) * Matern(nu=2.5) + WhiteKernel() [17] |
| **Acquisition Function** | LCB | Lower Confidence Bound: $LCB = \mu - 1.2\sigma$ ($\kappa = 1.2$) [17] |
| **Initial Exploration** | 10 samples | Purely random parameter sets evaluated to prime the GP model [17] |
| **Optimization Loops** | 40 iterations | Sequential exploitation steps guided by the LCB argmin surface |
| **Internal Search Strategy** | 500 candidates | Randomized candidate sets generated per step to minimize LCB [17] |
| **Search Bounds (Kp, Ki)** | $[0.000001, 0.01]$ | Uniform distribution sampling bounds |
| **Search Bounds (Kd)** | $[0.0, 0.01]$ | Uniform distribution sampling bounds |
| **Total Cost Budget** | 50 evaluations | Fixed budget convergence (10 init steps + 40 loop steps) |

#### 2.5 Optuna (Tree-structured Parzen Estimator)
* **Operational Context:** Density estimation-based automated sequential search framework.
* **Convergence Criteria:** Executes a strict sequential evaluation budget of exactly 100 trials.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Core Framework Engine** | Optuna Framework | Automated optimization management using optuna.create_study [18] |
| **Default Sampler Strategy** | Optuna TPE Sampler | Tree-structured Parzen Estimator kernel density algorithm [18] |
| **Evaluation Budget** | 100 Trials | Exhaustive density-based exploration trials [18] |
| **Search Bounds (Kp, Ki)** | $[0.00000001, 0.1]$ | Expansive continuous uniform float spaces (`trial.suggest_float`) |
| **Search Bounds (Kd)** | $[0.0, 0.1]$ | Expansive continuous uniform float spaces (`trial.suggest_float`) |
| **Total Cost Budget** | 100 evaluations | Fixed budget convergence criteria dictated by trial count |

#### 2.6 RL (Proximal Policy Optimization)
* **Operational Context:** Continuous policy-based online adaptation framework mapping live error states to gains.
* **Convergence Criteria:** Policy agent executes its learning updates for 5,000 timesteps per invocation block.

| Hyperparameter / Property | Value / Configuration | Literature Reference / Rationale |
| :--- | :--- | :--- |
| **Policy Architecture** | MlpPolicy | Actor-Critic Multi-Layer Perceptron architecture [19] |
| **State Space (S)** | $s_t = [e_t, e_{i,t}, e_{d,t}]^T \in \mathbb{R}^3$ | Continuous Gymnasium space tracking three localized PID errors |
| **Action Space (A)** | $a_t = [\Delta K_p, \Delta K_i, \Delta K_d]^T$ | Continuous adjustments bounded within $[-0.0005, 0.0005]^3$ |
| **Step Reward (R_t)** | $R_t = -\|u_t\|$ | Cost-driven negative penalty tracking absolute control effort |
| **Training Budget / Setup** | 5,000 timesteps | Total environment interaction steps per online training sequence [19] |
| **Learning Rate (eta)** | $0.0003$ | Canonical initial step optimizer Adam configuration [19] |
| **Training Procedure** | Online / Runtime | Trained dynamically on active data sliding batches [19] |
| **Hardware Device** | device="cpu" | Localized CPU constraint to ensure architecture-agnostic execution |
| **Operational Bounds** | $[0.0, 1.0]$ | Environment clipping thresholds enforced on absolute active gains |

---

### 3. Jitter Control & Post-Processing Safeguards
Stochastic optimizers can occasionally introduce noisy gain suggestions. To protect the physical plant from aggressive chattering and ensure closed-loop stability, every recommended parameter set passes through an intermediate Jitter Control filter:

* **Deadzone Threshold:** Updates smaller than $\delta < 10^{-6}$ are discarded to block numerical oscillations driven by measurement background noise.
* **Dynamic Saturation Cap:** The max rate of change between successive adjustments is clamped at $\Delta_{\text{max}} = 15\%$ relative to the active value to suppress sudden actuator spikes.
* **Exponential Smoothing Factors:** Valid filtered updates are passed through individual Exponential Moving Average (EMA) channels to ensure a bumpless transfer:
  * Proportional Gain Filter: $\alpha_p = 0.20$
  * Integral Gain Filter: $\alpha_i = 0.05$
  * Derivative Gain Filter: $\alpha_d = 0.40$

---

### 4. Code References (Academic Bibliography)
* **[14]** J. Spall, "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation," *IEEE Transactions on Automatic Control*, vol. 37, no. 3, pp. 332-341, 1992.
* **[15]** D. E. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*, 1st ed. USA: Addison-Wesley Longman Publishing Co., Inc., 1989.
* **[16]** J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proceedings of ICNN'95 - International Conference on Neural Networks*, vol. 4, 1995, pp. 1942-1948.
* **[17]** B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas, "Taking the human out of the loop: A review of bayesian optimization," *Proceedings of the IEEE*, vol. 104, no. 1, pp. 148-175, 2016.
* **[18]** T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2019, pp. 2623–2631.
* **[19]** J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.