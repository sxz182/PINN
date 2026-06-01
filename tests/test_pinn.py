import unittest

import numpy as np

from pinn import PINN1D


class TestPINN1D(unittest.TestCase):
    def test_training_reduces_data_error(self):
        x_data = np.linspace(0, 1, 10).reshape(-1, 1)
        y_data = np.exp(-x_data)
        x_phys = np.linspace(0, 1, 32).reshape(-1, 1)

        model = PINN1D(hidden_size=12, lambda_coeff=1.0, physics_weight=0.5, seed=7)
        initial_mse = np.mean((model.predict(x_data) - y_data) ** 2)
        history = model.train(
            x_data=x_data,
            y_data=y_data,
            x_phys=x_phys,
            epochs=800,
            learning_rate=0.01,
        )
        final_mse = np.mean((model.predict(x_data) - y_data) ** 2)

        self.assertGreater(history[0], history[-1])
        self.assertGreater(initial_mse, final_mse)


if __name__ == "__main__":
    unittest.main()
