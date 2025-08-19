"""
This script is an example of a regression task using the EDTS library.
"""
import numpy as np
import pandas as pd
from edts import EDTS
import logging


# Generate a synthetic regression dataset with a known equation
def generate_data(num_samples=10_000):
    x_0 = np.random.normal(0, 10, num_samples)
    x_1 = np.random.normal(0, 10, num_samples)

    # y = 2*x0 + 3*x1 + 5
    y = 2 * x_0 + 3 * x_1 + 5 + np.random.normal(0, 1, num_samples)

    return pd.DataFrame({'x0': x_0, 'x1': x_1, 'target': y})


if __name__ == "__main__":
    # Suppress JAX warnings
    logging.basicConfig(level=logging.INFO)
    jax_logger = logging.getLogger('jax')
    jax_logger.setLevel(logging.ERROR)

    data = generate_data()

    edts = EDTS(
        data=data,
        feature_names=['x0', 'x1'],
        building_blocks=[
            'c_',
            'c_*x_',
            'c_*x_*x_',
            'c_*x_**2',
            'sin(c_*x_)',
        ],
        iterations=2000,
    )

    edts.beam_search(
        beam_width=10,
        max_depth=4,
    )

    print("Best equation found:")
    print(edts.best_equation)
    print("True equation: 2*x0 + 3*x1 + 5")
