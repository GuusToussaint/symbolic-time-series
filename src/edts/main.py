"""EDTS - Equation Discovery for Time Series"""
import itertools
import concurrent.futures
import logging
import pandas as pd
import numpy as np
import jax.numpy as jnp
import sympy as sp
from edts.optimiser import Optimiser


class EDTS:
    """
    EDTS (Equation Discovery for Time Series) is a framework for
    discovering equations.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_names: list[str],
        building_blocks: list[str],
        iterations: int = 100,
    ):
        self.feature_names = feature_names
        self.logger = logging.getLogger("edc")
        self.n_cores = 5
        self.optimiser = Optimiser(
            var_values={
                name: jnp.asarray(data[name].values)
                for name in feature_names
            },
            target_values=jnp.asarray(data["target"].values),
            lr=5e-2,
        )
        self.best_equation = None
        self.best_loss = np.inf
        self.equation_counter = 0
        self.total_equations = 0
        self.iterations = iterations
        self._expand_building_blocks(building_blocks, len(feature_names))

    def _expand_building_blocks(self, building_blocks, num_features):
        def _expand_building_block(building_block, num_features):
            variables_in_building_block = building_block.count("x_")
            combinations = itertools.combinations(
                range(num_features), variables_in_building_block
            )

            current_building_blocks = []
            for combination in combinations:
                current_building_block = building_block
                for i in combination:
                    current_building_block = current_building_block.replace(
                        "x_", f"x{i}", 1
                    )
                current_building_blocks.append(current_building_block)

            return current_building_blocks

        expanded_building_blocks = []
        for bb in building_blocks:
            expanded_building_blocks.extend(
                _expand_building_block(bb, num_features)
            )

        self.logger.info(
            "Number of expanded building blocks: %d",
            len(expanded_building_blocks)
        )
        self.building_blocks = expanded_building_blocks

    def _get_loss_and_config(self, equation, parallel=False):
        potential_equation_expanded = sp.Matrix(equation).T * sp.Matrix(
            self.building_blocks
        )
        equation_sympy = potential_equation_expanded[0]
        equation_str = self._replace_constants(str(equation_sympy))
        equation_sympy = sp.sympify(equation_str)

        loss, config = self.optimiser.optimise(equation_sympy, self.iterations)
        equation_with_constants = equation_sympy.subs(config)
        self.logger.debug(
            "Equation: %s, Loss: %s", str(equation_with_constants), loss
        )

        if parallel:
            return loss, equation_with_constants

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_equation = equation_with_constants
        if (self.equation_counter % 10) == 0:
            self.logger.info(
                "Best loss at iteration %d: %s",
                self.equation_counter,
                self.best_loss,
            )

        self.equation_counter += 1
        return loss, config

    def _get_losses_and_configs_parallel(self, equations):
        # Setup the pool of workers
        self.equation_counter = 0
        self.total_equations = len(equations)
        executer = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_cores
        )
        futures = []
        results = []

        def done_callback(arg):
            res = arg.result()
            loss, equation = res
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_equation = equation
            self.equation_counter += 1
            if (self.equation_counter % 10) == 0:
                self.logger.info(
                    "Best loss at iteration %d/%d: %s",
                    self.equation_counter,
                    self.total_equations,
                    self.best_loss
                )

        for equation in equations:
            f = executer.submit(
                self._get_loss_and_config,
                equation,
                parallel=True
            )
            f.add_done_callback(done_callback)
            futures.append(f)

        concurrent.futures.wait(futures)
        for f in futures:
            results.append(f.result())

        return results

    def _replace_constants(self, equation: str):
        constants_counter = 0
        while equation.count("c_"):
            equation = equation.replace("c_", f"c{constants_counter}", 1)
            constants_counter += 1
        return equation

    def beam_search(self, beam_width, max_depth):
        """Performs a beam serch to find the best equation."""
        n_blocks = len(self.building_blocks)
        b_matrix = np.identity(n_blocks, dtype=np.uint8)
        i_matrix = np.identity(n_blocks, dtype=np.uint8)

        for depth in range(min(n_blocks, max_depth)):
            if b_matrix.shape[0] == 0:
                break

            losses_and_configs = self._get_losses_and_configs_parallel(
                b_matrix
            )
            losses = [loss for loss, _ in losses_and_configs]

            relevant_losses = np.argsort(losses)[:beam_width]
            self.logger.info(
                "Best loss at depth %d: %s", depth, losses[relevant_losses[0]]
            )

            beam_matrixes = []
            for current_loss in relevant_losses:
                current_loss_row = b_matrix[current_loss, :]
                current_loss_indexes = np.where(current_loss_row == 1)[0]

                # create a matrix of the current loss row
                current_loss_matrix = np.array(
                    [current_loss_row] * b_matrix.shape[0]
                ).reshape(i_matrix.shape)

                # Multiply with the identity matrix
                current_loss_matrix = current_loss_matrix | i_matrix

                # Remove the current loss rows
                current_loss_matrix_clean = np.delete(
                    current_loss_matrix, current_loss_indexes, 0
                )

                # Add the current beam matrix
                beam_matrixes.append(current_loss_matrix_clean)

            b_matrix = np.vstack(beam_matrixes)
            b_matrix = np.unique(b_matrix, axis=0)
