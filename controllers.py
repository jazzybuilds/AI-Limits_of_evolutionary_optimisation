"""
Neural network controllers for neuroevolution.

Supports:
  - FeedforwardController: obs -> tanh hidden -> linear output
  - CTRNNController: continuous-time recurrent NN with Euler integration

Both share a common interface:
    ctrl.reset()
    action = ctrl.forward(obs)

Parameters are set via ctrl.set_params(params_dict) where params_dict
comes from GenomeDecoder.decode().
"""
import numpy as np


class FeedforwardController:
    """
    Two-layer feedforward network.

    Architecture:  obs -> W1/b1 -> tanh -> W2/b2 -> output

    Params dict keys: W1, b1, W2, b2
    """

    def __init__(self, obs_dim, act_dim, hidden_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        # Initialise to zeros; will be overwritten by set_params
        self.W1 = np.zeros((hidden_size, obs_dim))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.zeros((act_dim, hidden_size))
        self.b2 = np.zeros(act_dim)

    def set_params(self, params):
        """Load decoded parameters from genome."""
        self.W1 = params['W1'].reshape(self.hidden_size, self.obs_dim)
        self.b1 = params['b1']
        self.W2 = params['W2'].reshape(self.act_dim, self.hidden_size)
        self.b2 = params['b2']

    def reset(self):
        """No internal state to reset for feedforward."""
        pass

    def forward(self, obs):
        """
        Args:
            obs: numpy array of shape (obs_dim,)

        Returns:
            output: numpy array of shape (act_dim,)
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()
        h = np.tanh(self.W1 @ obs + self.b1)
        out = self.W2 @ h + self.b2
        return out


class CTRNNController:
    """
    Continuous-Time Recurrent Neural Network (Beer 1995).

    Dynamics (Euler integration):
        tau_i * dy_i/dt = -y_i + sum_j(W_ij * sigma(y_j + bias_j)) + sum_k(W_in_ik * x_k)
        y_i(t+dt) = y_i(t) + dt/tau_i * (-y_i(t) + ...)

    Output:
        out = W_out @ sigma(y + bias)

    where sigma = tanh.

    Params dict keys: W, W_in, W_out, bias, tau
    """

    def __init__(self, obs_dim, act_dim, hidden_size, dt=0.2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.dt = dt

        # Network parameters (overwritten by set_params)
        self.W     = np.zeros((hidden_size, hidden_size))
        self.W_in  = np.zeros((hidden_size, obs_dim))
        self.W_out = np.zeros((act_dim, hidden_size))
        self.bias  = np.zeros(hidden_size)
        self.tau   = np.ones(hidden_size)

        # Internal state
        self.y = np.zeros(hidden_size)

    def set_params(self, params):
        """Load decoded parameters from genome."""
        self.W     = params['W'].reshape(self.hidden_size, self.hidden_size)
        self.W_in  = params['W_in'].reshape(self.hidden_size, self.obs_dim)
        self.W_out = params['W_out'].reshape(self.act_dim, self.hidden_size)
        self.bias  = params['bias']
        self.tau   = np.clip(params['tau'], 0.01, None)  # tau > 0

    def reset(self):
        """Reset internal state to zero between episodes."""
        self.y = np.zeros(self.hidden_size)

    def forward(self, obs):
        """
        Run one timestep.

        Args:
            obs: numpy array of shape (obs_dim,)

        Returns:
            output: numpy array of shape (act_dim,)
        """
        obs = np.asarray(obs, dtype=np.float32).flatten()
        activation = np.tanh(self.y + self.bias)
        dy = (-self.y
              + self.W @ activation
              + self.W_in @ obs)
        self.y = self.y + (self.dt / self.tau) * dy
        out = self.W_out @ np.tanh(self.y + self.bias)
        return out


def build_controller(controller_type, obs_dim, act_dim, hidden_size, dt=0.2):
    """
    Factory: returns the appropriate controller instance.

    Args:
        controller_type: 'feedforward' or 'ctrnn'
        obs_dim: observation input dimension
        act_dim: action output dimension
        hidden_size: hidden / recurrent units
        dt: integration timestep (CTRNN only)

    Returns:
        FeedforwardController or CTRNNController
    """
    if controller_type == 'feedforward':
        return FeedforwardController(obs_dim, act_dim, hidden_size)
    elif controller_type == 'ctrnn':
        return CTRNNController(obs_dim, act_dim, hidden_size, dt=dt)
    else:
        raise ValueError(f"Unknown controller_type '{controller_type}'. "
                         "Choose 'feedforward' or 'ctrnn'.")
