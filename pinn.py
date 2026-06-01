import numpy as np


def _zero_forcing(x):
    return np.zeros_like(x)


class PINN1D:
    def __init__(self, hidden_size=20, lambda_coeff=1.0, physics_weight=1.0, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.lambda_coeff = float(lambda_coeff)
        self.physics_weight = float(physics_weight)
        self.W1 = rng.normal(0.0, 0.5, size=(1, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0.0, 0.5, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def _as_column(x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        return x

    def _forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        y = a1 @ self.W2 + self.b2
        return z1, a1, y

    def predict(self, x):
        x = self._as_column(x)
        _, _, y = self._forward(x)
        return y

    def _physics_residual(self, x_phys, forcing_fn):
        z1, a1, y = self._forward(x_phys)
        da1_dz1 = 1.0 - a1**2
        dy_dx = np.sum(da1_dz1 * self.W1 * self.W2.T, axis=1, keepdims=True)
        forcing = forcing_fn(x_phys)
        residual = dy_dx + self.lambda_coeff * y - forcing
        return residual, z1, a1, y, da1_dz1

    def train(
        self,
        x_data,
        y_data,
        x_phys,
        forcing_fn=None,
        epochs=2000,
        learning_rate=1e-2,
        verbose=False,
    ):
        x_data = self._as_column(x_data)
        y_data = self._as_column(y_data)
        x_phys = self._as_column(x_phys)

        if forcing_fn is None:
            forcing_fn = _zero_forcing

        history = []
        for epoch in range(epochs):
            z1_d, a1_d, y_pred_d = self._forward(x_data)
            err_d = y_pred_d - y_data
            data_loss = np.mean(err_d**2)

            grad_y = (2.0 / len(x_data)) * err_d
            dW2_data = a1_d.T @ grad_y
            db2_data = np.sum(grad_y, axis=0, keepdims=True)
            da1_data = grad_y @ self.W2.T
            dz1_data = da1_data * (1.0 - a1_d**2)
            dW1_data = x_data.T @ dz1_data
            db1_data = np.sum(dz1_data, axis=0, keepdims=True)

            residual, z1_p, a1_p, y_pred_p, da1_dz1_p = self._physics_residual(x_phys, forcing_fn)
            phys_loss = np.mean(residual**2)

            r_factor = (2.0 / len(x_phys)) * residual
            weighted_activation_grad = self.W2.T * da1_dz1_p
            activation_second_deriv_factor = -2.0 * a1_p * da1_dz1_p

            dr_dW2 = da1_dz1_p * self.W1 + self.lambda_coeff * a1_p
            dr_db2 = np.full_like(residual, self.lambda_coeff)
            dr_db1 = self.W2.T * (
                activation_second_deriv_factor * self.W1 + self.lambda_coeff * da1_dz1_p
            )
            dr_dW1 = (
                self.W2.T * da1_dz1_p
                + self.W2.T * self.W1 * activation_second_deriv_factor * x_phys
                + self.lambda_coeff * weighted_activation_grad * x_phys
            )

            dW2_phys = np.sum(r_factor * dr_dW2, axis=0, keepdims=True).T
            db2_phys = np.array([[np.sum(r_factor * dr_db2)]])
            db1_phys = np.sum(r_factor * dr_db1, axis=0, keepdims=True)
            dW1_phys = np.sum(r_factor * dr_dW1, axis=0, keepdims=True)

            dW2 = dW2_data + self.physics_weight * dW2_phys
            db2 = db2_data + self.physics_weight * db2_phys
            dW1 = dW1_data + self.physics_weight * dW1_phys
            db1 = db1_data + self.physics_weight * db1_phys

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            total_loss = data_loss + self.physics_weight * phys_loss
            history.append(float(total_loss))
            if verbose and (epoch + 1) % 200 == 0:
                print(f"epoch={epoch + 1}, loss={total_loss:.6f}")

        return history


def demo():
    x_data = np.linspace(0, 1, 12).reshape(-1, 1)
    y_data = np.exp(-x_data)
    x_phys = np.linspace(0, 1, 64).reshape(-1, 1)

    model = PINN1D(hidden_size=16, lambda_coeff=1.0, physics_weight=0.5, seed=1)
    initial_pred = model.predict(x_data)
    initial_mse = np.mean((initial_pred - y_data) ** 2)

    history = model.train(
        x_data=x_data,
        y_data=y_data,
        x_phys=x_phys,
        epochs=1500,
        learning_rate=0.01,
    )

    final_pred = model.predict(x_data)
    final_mse = np.mean((final_pred - y_data) ** 2)
    print(f"initial_mse={initial_mse:.6f}, final_mse={final_mse:.6f}, final_loss={history[-1]:.6f}")


if __name__ == "__main__":
    demo()
