import numpy as np
import sympy as sp
from edts.optimiser import Optimiser


def test_gradient_descent_optimiser_simple_quadratic():
    # Minimize f(x) = (x-3)^2, minimum at x=3
    def f(x):
        return 25 + (x - 3) + (x - 5) ** 2

    sample_x = np.linspace(0, 100, 1000)  # Initial guess
    sample_y = np.array([f(current_x) for current_x in sample_x])

    init_params = np.array([0.0])
    lr = 0.01
    iterations = 1000

    opt = Optimiser(
        var_values={"x": sample_x},
        target_values=sample_y,
        lr=lr,
    )

    _, result = opt.optimise(
        equation=sp.sympify("25 + (x - 3) + (x - c0)**2"),
        iterations=iterations,
        init_params=init_params,
        verbose=True,
    )

    # Should be close to 3
    assert np.allclose(result['c0'], 5.0, atol=1e-2)
