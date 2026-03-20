"""
Fitness Landscape Adapter for Stage 1 Neuroevolution.

Acts as a drop-in replacement for NKLandscape so the existing
GeneticAlgorithm class works unchanged.

Supports two evaluation environments:
  - 'minigrid': MiniGrid tasks (Chevalier-Boisvert et al.)
  - 'cartpole': CartPole-v1 (Gymnasium)

Fitness = w_perf * train_performance + w_gen * generalisation_performance

train_performance:  mean success/return over TRAIN_SEEDS mazes / CartPole rollouts
generalisation:     mean success/return over TEST_SEEDS  (unseen mazes)

Note: learnability is parked until Stage 2.
"""
import numpy as np
import gymnasium as gym
try:
    import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium
except ModuleNotFoundError:
    pass  # minigrid optional; only needed for env_type='minigrid'

from genome_decoder import build_decoder_for_controller
from controllers import build_controller


# ---------------------------------------------------------------------------
# Helper: run one episode, return total reward and success flag
# ---------------------------------------------------------------------------

def _run_episode(env, controller, max_steps, obs_flat_fn, seed=None,
                 progress_fn=None, continuous_action=False):
    """
    Roll out one episode.

    Args:
        env: Gymnasium environment
        controller: FeedforwardController or CTRNNController
        max_steps: episode step limit
        obs_flat_fn: callable(obs) -> 1D numpy float array
        seed: optional integer seed for this episode
        progress_fn: optional callable(obs) -> float to track episode progress.
            If provided, the max value seen is returned as max_progress.
        continuous_action: if True, pass tanh(logits) directly as action
            (for Box action spaces); if False, use argmax (Discrete).

    Returns:
        (total_reward: float, success: bool, max_progress: float)
        max_progress is -inf when progress_fn is None.
    """
    obs, _ = env.reset(seed=seed)
    controller.reset()
    total_reward = 0.0
    success = False
    max_progress = -np.inf

    if progress_fn is not None:
        max_progress = progress_fn(obs)

    for _ in range(max_steps):
        flat_obs = obs_flat_fn(obs)
        logits = controller.forward(flat_obs)
        if continuous_action:
            action = np.tanh(logits)   # maps R -> [-1, 1], matches Box action space
        else:
            action = int(np.argmax(logits))

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if progress_fn is not None:
            max_progress = max(max_progress, progress_fn(obs))

        if terminated:
            # MiniGrid sets reward > 0 on goal; CartPole terminates on failure
            success = reward > 0.0
            break
        if truncated:
            break

    return total_reward, success, max_progress


# ---------------------------------------------------------------------------
# Observation flatteners
# ---------------------------------------------------------------------------

def _minigrid_flat(obs):
    """Flatten MiniGrid dict observation image to 1D float array."""
    if isinstance(obs, dict):
        img = obs.get('image', obs.get('obs', list(obs.values())[0]))
    else:
        img = obs
    return np.asarray(img, dtype=np.float32).flatten() / 10.0   # normalise


def _cartpole_flat(obs):
    """CartPole observation is already a 1D array."""
    return np.asarray(obs, dtype=np.float32)


def _acrobot_flat(obs):
    """Acrobot obs: [cos θ1, sin θ1, cos θ2, sin θ2, dθ1/dt, dθ2/dt] — already flat."""
    return np.asarray(obs, dtype=np.float32)


def _acrobot_progress(obs):
    """
    Dense progress signal for Acrobot: -cos(θ1) - cos(θ1+θ2).

    Values:
        ~-2.0  : start (both links hanging straight down)
        > 1.0  : goal reached (tip of link 2 above threshold = termination)
        ~+2.0  : fully inverted

    Using obs layout: [cos θ1, sin θ1, cos θ2, sin θ2, dθ1, dθ2]
    cos(θ1+θ2) = cos(θ1)cos(θ2) - sin(θ1)sin(θ2)
    """
    c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    return -c1 - (c1 * c2 - s1 * s2)


def _mountaincar_progress(obs):
    """
    Dense progress signal for MountainCar: car's x-position.

    Values:
        ~-0.5  : start
        >= 0.45: goal
        range  : [-1.2, 0.6]
    """
    return float(obs[0])


def _make_acrobot(params: dict):
    """
    Create an Acrobot-v1 env and apply physics parameter overrides.

    Supported keys (all optional, defaults in brackets):
        LINK_LENGTH_1  - length of link 1   (default 1.0)
        LINK_LENGTH_2  - length of link 2   (default 1.0)
        LINK_MASS_1    - mass of link 1     (default 1.0)
        LINK_MASS_2    - mass of link 2     (default 1.0)
        LINK_COM_POS_1 - CoM pos of link 1  (default 0.5)
        LINK_COM_POS_2 - CoM pos of link 2  (default 0.5)
        LINK_MOI       - moment of inertia  (default 1.0)
    """
    env = gym.make('Acrobot-v1')
    u = env.unwrapped
    for key, val in params.items():
        setattr(u, key, val)
    return env


def _lunarlander_flat(obs):
    """LunarLander obs: 8D — already flat."""
    return np.asarray(obs, dtype=np.float32)


def _lunarlander_progress(obs):
    """
    Dense landing-quality proxy for LunarLander.

    Always rewards: being centred (x≈0), low lateral velocity (vx≈0),
    controlled sink rate (vy≈0), upright posture (angle≈0).

    Descent reward (low y) is GATED by horizontal proximity to the pad:
    full reward at x=0, tapering to zero at |x|=0.5.
    This prevents the agent learning to crash away from the target zone.

    Range: ~-3 (start: high, drifting far) to +2 (both legs down, all zero).
    Scoring: clip((max_progress + 3) / 5, 0, 1)
    """
    x, y, vx, vy, angle, ang_vel, leg_l, leg_r = [float(o) for o in obs]
    over_pad = max(0.0, 1.0 - abs(x) * 2.0)   # 1.0 at x=0, 0.0 at |x|>=0.5
    return (
        - abs(x)          # reward centering over pad
        - abs(vx)         # reward low lateral velocity
        - abs(vy)         # reward controlled vertical velocity
        - abs(angle)      # reward upright posture
        - y * over_pad    # reward low altitude ONLY when near the pad
        + leg_l + leg_r   # touchdown bonus
    )


def _make_lunarlander(params: dict):
    """
    Create a LunarLander-v3 env with parameter overrides.

    Supported keys (passed as kwargs to gym.make):
        gravity        (default -10.0)
        wind_power     (default 0.0)
        turbulence_power (default 0.0)
        enable_wind    (default False)
    """
    kwargs = {k: v for k, v in params.items()}
    return gym.make('LunarLander-v3', **kwargs)


def _make_lunarlander_continuous(params: dict):
    """
    Create a LunarLanderContinuous-v3 env with parameter overrides.
    Same physics kwargs as LunarLander-v3; action space is Box(2,) instead of Discrete(4).
    """
    kwargs = {k: v for k, v in params.items()}
    return gym.make('LunarLanderContinuous-v3', **kwargs)


def _mountaincar_flat(obs):
    """MountainCar obs: [position, velocity] — already flat."""
    return np.asarray(obs, dtype=np.float32)


def _make_mountaincar(params: dict):
    """
    Create a MountainCar-v0 env and apply physics overrides.

    Supported keys (all optional):
        min_position   (default -1.2)
        max_position   (default 0.6)
        max_speed      (default 0.07)
        goal_position  (default 0.45)
        power          (default 0.001)
    """
    env = gym.make('MountainCar-v0')
    u = env.unwrapped
    for key, val in params.items():
        setattr(u, key, val)
    return env


def _make_mountaincarcontinuous(params: dict):
    """
    Create a MountainCarContinuous-v0 env and apply physics overrides.

    Supported keys (all optional, same as MountainCar-v0):
        min_position, max_position, max_speed, goal_position, power
    """
    env = gym.make('MountainCarContinuous-v0')
    u = env.unwrapped
    for key, val in params.items():
        setattr(u, key, val)
    return env


def _make_cartpole(params: dict):
    """
    Create a CartPole-v1 env and apply physics parameter overrides.

    Supported keys (all optional):
        length     - half-length of pole (default 0.5)
        masspole   - mass of pole        (default 0.1)
        masscart   - mass of cart        (default 1.0)
        gravity    - gravitational accel (default 9.8)
        force_mag  - force applied       (default 10.0)
    """
    env = gym.make('CartPole-v1')
    u = env.unwrapped
    if 'length'   in params: u.length   = params['length']
    if 'masspole' in params: u.masspole = params['masspole']
    if 'masscart' in params: u.masscart = params['masscart']
    if 'gravity'  in params: u.gravity  = params['gravity']
    if 'force_mag' in params: u.force_mag = params['force_mag']
    # Keep derived quantities consistent
    u.total_mass      = u.masscart + u.masspole
    u.polemass_length = u.masspole * u.length
    return env


# ---------------------------------------------------------------------------
# Main landscape adapter
# ---------------------------------------------------------------------------

class NeuroevoFitnessLandscape:
    """
    Drop-in NKLandscape replacement for neuroevolution on MiniGrid / CartPole.

    The existing GeneticAlgorithm only uses:
        landscape.N          -> genome length
        landscape.evaluate(genome) -> float fitness

    Both are provided here.

    Args:
        config: dict of experiment configuration (from run_stage1.py)
    """

    def __init__(self, config):
        self.config = config
        env_type        = config['ENV_TYPE']          # 'minigrid' or 'cartpole'
        controller_type = config['CONTROLLER_TYPE']   # 'feedforward' or 'ctrnn'
        hidden_size     = config['HIDDEN_SIZE']
        bits_per_weight = config['BITS_PER_WEIGHT']

        # ---- Resolve observation and action dimensions ----
        obs_dim, act_dim = self._probe_dims(config)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # ---- Build genome decoder ----
        self.decoder = build_decoder_for_controller(
            controller_type, obs_dim, act_dim,
            hidden_size, bits_per_weight
        )
        # Genome encoding: 'binary' (default) or 'realvalued'.
        self.genome_encoding = config.get('GENOME_ENCODING', 'binary')

        # Apply neutrality: expand genome with neutral padding bits.
        # With p=0.75, only 25% of bits encode weights; 75% are neutral.
        # This directly mirrors the NKp landscape p parameter.
        # Neutrality is ignored for real-valued encoding (no padding concept).
        p = config.get('NEUTRALITY_P', 0.0)
        active_bits = self.decoder.N
        if self.genome_encoding == 'realvalued':
            self.N = self.decoder.N_params   # one float per weight, no bit packing
        elif p > 0.0:
            self.N = max(active_bits, int(round(active_bits / (1.0 - p))))
        else:
            self.N = active_bits   # required by GeneticAlgorithm

        # ---- Build prototype controller (reused per evaluation) ----
        self.controller = build_controller(
            controller_type, obs_dim, act_dim, hidden_size,
            dt=config.get('CTRNN_DT', 0.2)
        )

        # ---- Environment factories ----
        self.env_type = env_type
        if env_type == 'minigrid':
            self.obs_flat_fn  = _minigrid_flat
            self.train_envs   = config['MINIGRID_TRAIN_ENVS']
            self.test_envs    = config['MINIGRID_TEST_ENVS']
            self.max_steps    = config['MAX_STEPS_PER_EPISODE']
        elif env_type == 'cartpole':
            self.obs_flat_fn  = _cartpole_flat
            self.train_envs   = config.get('CARTPOLE_TRAIN_PARAMS', [{}])
            self.test_envs    = config.get('CARTPOLE_TEST_PARAMS',  [{}])
            self.max_steps    = config['MAX_STEPS_PER_EPISODE']
        elif env_type == 'acrobot':
            self.obs_flat_fn  = _acrobot_flat
            self.train_envs   = config.get('ACROBOT_TRAIN_PARAMS', [{}])
            self.test_envs    = config.get('ACROBOT_TEST_PARAMS',  [{}])
            self.max_steps    = config.get('ACROBOT_MAX_STEPS', config['MAX_STEPS_PER_EPISODE'])
        elif env_type == 'lunarlander':
            self.obs_flat_fn  = _lunarlander_flat
            self.train_envs   = config.get('LUNARLANDER_TRAIN_PARAMS', [{}])
            self.test_envs    = config.get('LUNARLANDER_TEST_PARAMS',  [{}])
            self.max_steps    = config['MAX_STEPS_PER_EPISODE']
        elif env_type == 'lunarlander_continuous':
            self.obs_flat_fn  = _lunarlander_flat
            self.train_envs   = config.get('LUNARLANDER_CONTINUOUS_TRAIN_PARAMS',
                                           config.get('LUNARLANDER_TRAIN_PARAMS', [{}]))
            self.test_envs    = config.get('LUNARLANDER_CONTINUOUS_TEST_PARAMS',
                                           config.get('LUNARLANDER_TEST_PARAMS', [{}]))
            self.max_steps    = config['MAX_STEPS_PER_EPISODE']
        elif env_type == 'mountaincar':
            self.obs_flat_fn  = _mountaincar_flat
            self.train_envs   = config.get('MOUNTAINCAR_TRAIN_PARAMS', [{}])
            self.test_envs    = config.get('MOUNTAINCAR_TEST_PARAMS',  [{}])
            self.max_steps    = config.get('MOUNTAINCAR_MAX_STEPS', config['MAX_STEPS_PER_EPISODE'])
        elif env_type == 'mountaincarcontinuous':
            self.obs_flat_fn  = _mountaincar_flat   # same 2D obs: [position, velocity]
            self.train_envs   = config.get('MOUNTAINCARCONTINUOUS_TRAIN_PARAMS', [{}])
            self.test_envs    = config.get('MOUNTAINCARCONTINUOUS_TEST_PARAMS',  [{}])
            self.max_steps    = config.get('MOUNTAINCARCONTINUOUS_MAX_STEPS',
                                           config['MAX_STEPS_PER_EPISODE'])
        else:
            raise ValueError(f"Unknown ENV_TYPE '{env_type}'. "
                             "Choose 'cartpole', 'acrobot', 'lunarlander', "
                             "'lunarlander_continuous', 'mountaincar', "
                             "'mountaincarcontinuous', or 'minigrid'.")

        # Continuous action spaces (Box) require tanh output instead of argmax
        self.continuous_actions = env_type in ('mountaincarcontinuous', 'lunarlander_continuous')

        self.train_seeds = config['TRAIN_SEEDS']
        self.test_seeds  = config['TEST_SEEDS']
        self.w_perf      = config['FITNESS_W_PERF']
        self.w_gen       = config['FITNESS_W_GEN']
        self.n_eval_eps  = config['N_EVAL_EPISODES']

        # Pre-create environments once; reused across all evaluations to
        # avoid repeated gym.make / env.close overhead (~6 creates per eval).
        self._train_env_list = [self._make_env_from_spec(s) for s in self.train_envs]
        self._test_env_list  = [self._make_env_from_spec(s) for s in self.test_envs]
        # Cache: genome.tobytes() -> (fitness, train_score, test_score)
        self._eval_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, genome):
        """
        Decode genome, evaluate on train + test environments, return fitness.

        Args:
            genome: binary numpy array of length self.N

        Returns:
            float in [0, 1]
        """
        key = genome.tobytes()
        if key not in self._eval_cache:
            if self.genome_encoding == 'realvalued':
                params = self.decoder.decode_realvalued(genome)
            else:
                params = self.decoder.decode(genome)
            self.controller.set_params(params)
            train_score = self._eval_envs(self._train_env_list, self.train_seeds)
            test_score  = self._eval_envs(self._test_env_list,  self.test_seeds)
            fitness = float(np.clip(
                self.w_perf * train_score + self.w_gen * test_score, 0.0, 1.0))
            self._eval_cache[key] = (fitness, float(train_score), float(test_score))
        return self._eval_cache[key][0]

    def evaluate_split(self, genome):
        """
        Same as evaluate() but returns (train_score, test_score, composite).
        Used for per-generation logging of the best individual.
        Reuses cached result from evaluate() — no extra episode rollouts.
        """
        key = genome.tobytes()
        if key not in self._eval_cache:
            self.evaluate(genome)  # populate cache
        fitness, train_score, test_score = self._eval_cache[key]
        return train_score, test_score, fitness

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_envs(self, env_list, seeds):
        """
        Evaluate controller across pre-created environments and seeds.

        env_list: list of gym environments (pre-created in __init__)

        Returns normalised mean score in [0, 1].
        """
        _progress_fn = {
            'acrobot':                _acrobot_progress,
            'mountaincar':            _mountaincar_progress,
            'mountaincarcontinuous':  _mountaincar_progress,  # same position proxy
            # lunarlander uses cumulative reward directly — no progress proxy needed
        }.get(self.env_type)

        scores = []
        for env in env_list:
            ep_scores = []
            for seed in seeds:
                for ep_i in range(self.n_eval_eps):
                    ep_seed = seed if ep_i == 0 else None
                    reward, success, max_progress = _run_episode(
                        env, self.controller,
                        self.max_steps, self.obs_flat_fn,
                        seed=ep_seed, progress_fn=_progress_fn,
                        continuous_action=self.continuous_actions
                    )
                    ep_scores.append(
                        self._score_episode(reward, success, max_progress)
                    )
            scores.append(np.mean(ep_scores))
        return float(np.mean(scores)) if scores else 0.0

    def _make_env_from_spec(self, spec):
        """Create env from a spec (dict of params or string env name)."""
        if isinstance(spec, dict):
            return self._make_env(spec)
        return gym.make(spec)

    def _make_env(self, params: dict):
        """Create the correct env type from a physics params dict."""
        if self.env_type == 'cartpole':
            return _make_cartpole(params)
        elif self.env_type == 'acrobot':
            return _make_acrobot(params)
        elif self.env_type == 'lunarlander':
            return _make_lunarlander(params)
        elif self.env_type == 'lunarlander_continuous':
            return _make_lunarlander_continuous(params)
        elif self.env_type == 'mountaincar':
            return _make_mountaincar(params)
        elif self.env_type == 'mountaincarcontinuous':
            return _make_mountaincarcontinuous(params)
        else:
            return gym.make(params)  # fallback: params is an env name string

    def _score_episode(self, reward: float, success: bool,
                       max_progress: float = -np.inf) -> float:
        """Normalise episode outcome to [0, 1] depending on env type."""
        if self.env_type == 'minigrid':
            return float(success)
        elif self.env_type == 'cartpole':
            # CartPole: +1/step reward, max = max_steps; rewards are always >= 0
            return float(np.clip(reward, 0, self.max_steps) / self.max_steps)
        elif self.env_type in ('lunarlander', 'lunarlander_continuous'):
            # Cumulative env reward: crash ≈ -100, good landing ≈ +200.
            # clip((reward + 100) / 300, 0, 1) → crash=0.0, landing=1.0.
            return float(np.clip((reward + 100.0) / 300.0, 0.0, 1.0))
        elif self.env_type == 'acrobot':
            # Dense height-proxy score: maps [-2, +2] → [0, 1].
            # Start (hanging) ≈ 0.0; goal threshold (1.0) → 0.75; full swing → 1.0
            # This gives gradient even when no agent has ever solved the task.
            return float(np.clip((max_progress + 2.0) / 4.0, 0.0, 1.0))
        elif self.env_type in ('mountaincar', 'mountaincarcontinuous'):
            # Dense position-proxy: maps [-1.2, 0.6] → [0, 1]
            return float(np.clip((max_progress + 1.2) / 1.8, 0.0, 1.0))
        return float(np.clip(reward / self.max_steps, 0, 1))

    @staticmethod
    def _probe_dims(config):
        """
        Create a temporary env to read obs_dim and act_dim automatically.
        Avoids hardcoding dimensions per environment.
        """
        env_type = config['ENV_TYPE']
        _probe_map = {
            'cartpole':               ('CartPole-v1',              None),
            'acrobot':                ('Acrobot-v1',               None),
            'lunarlander':            ('LunarLander-v3',              None),
            'lunarlander_continuous': ('LunarLanderContinuous-v3',    None),
            'mountaincar':            ('MountainCar-v0',              None),
            'mountaincarcontinuous':  ('MountainCarContinuous-v0', None),
            'minigrid':               (config['MINIGRID_TRAIN_ENVS'][0], _minigrid_flat),
        }
        probe_name, flat_fn = _probe_map.get(
            env_type, ('CartPole-v1', None))

        env = gym.make(probe_name)
        obs, _ = env.reset()

        if flat_fn is not None:
            obs_dim = flat_fn(obs).shape[0]
        else:
            obs_dim = env.observation_space.shape[0]

        # Discrete envs have action_space.n; continuous have action_space.shape
        if hasattr(env.action_space, 'n'):
            act_dim = env.action_space.n
        else:
            act_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, act_dim
