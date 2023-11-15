import numpy as np


def relu(x):
    return x * (x > 0)


class ANN:
    def __init__(self, input_dim, hidden_dim, output_dim, f=relu):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.f = f
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def init(self):
        input_dim, hidden_dim, output_dim = self.input_dim, self.hidden_dim, self.output_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim)

    def forward(self, x, action_max):
        z = self.f(x.dot(self.W1) + self.b1)
        return np.tanh(z.dot(self.W2) + self.b2) * action_max

    def sample_action(self, x, action_max):
        # assume input is a single state of size (D,)
        # first make it (N, D) to fit ML conventions
        x = np.atleast_2d(x)
        pred = self.forward(x, action_max)[0]
        # return np.random.choice(len(p), p=p)
        return pred

    def get_params(self):
        # return a flat array of parameters
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }

    def set_params(self, params):
        # params is a flat list
        # unflatten into individual weights
        input_dim, hidden_dim, output_dim = self.input_dim, self.hidden_dim, self.output_dim
        self.W1 = params[:input_dim * hidden_dim].reshape(input_dim, hidden_dim)
        self.b1 = params[input_dim * hidden_dim:input_dim * hidden_dim + hidden_dim]
        self.W2 = params[
                  input_dim * hidden_dim + hidden_dim:input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim].reshape(
            hidden_dim, output_dim)
        self.b2 = params[-output_dim:]
