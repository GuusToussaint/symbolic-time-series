import logging
import pandas as pd
import numpy as np
import sympy as sp
import itertools
from optimiser import Optimiser
import concurrent.futures

class EDTS:
    """
        EDTS (Equation Discovery for Time Series) is a framework for discovering equations
        that describe the relationships between features in time series data.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            feature_names: list[str],
            building_blocks: list[str],
            iterations: int = 100,
        ):
        self.feature_names = feature_names
        self.logger = logging.getLogger('edc')
        self.n_cores = 5
        self.optimiser = Optimiser(
            var_values={name: data[name].values for name in feature_names},
            y=data['target'].values,
            lr=5e-2,
        )
        self.best_loss = np.inf
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
            expanded_building_blocks.extend(_expand_building_block(bb, num_features))

        self.logger.info(f"Number of expanded building blocks: {len(expanded_building_blocks)}")
        self.building_blocks = expanded_building_blocks

    def _get_loss_and_config(self, equation, parallel=False):
        potential_equation_expanded = sp.Matrix(equation).T * sp.Matrix(self.building_blocks)
        equation_sympy = potential_equation_expanded[0]
        equation_sympy = sp.sympify(self._replace_constants(str(equation_sympy)))

        loss, config = self.optimiser.optimise(equation_sympy, self.iterations)
        equation_with_constants = equation_sympy.subs(config)
        self.logger.debug(f"Equation: {equation_with_constants}, Loss: {loss}")

        if parallel:
            return loss, equation_with_constants

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_equation = equation_with_constants
        if (self.equation_counter % 10) == 0:
            self.logger.info(f"Best loss at iteration {self.equation_counter}: {self.best_loss}")

        self.equation_counter += 1
        return loss, config

    def _get_losses_and_configs_parallel(self, equations):
        # Setup the pool of workers
        self.equation_counter = 0
        self.total_equations = len(equations)
        executer = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_cores)
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
                self.logger.info(f"Best loss at iteration {self.equation_counter}/{self.total_equations}: {self.best_loss}")

        for equation in equations:
            f = executer.submit(self._get_loss_and_config, equation, parallel=True)
            f.add_done_callback(done_callback)
            futures.append(f)
        
        concurrent.futures.wait(futures)
        for f in futures:
            results.append(f.result())

        return results

    def _replace_constants(self, equation: str):
        constants_counter = 0
        while equation.count("c_"):
            equation = equation.replace(
                "c_", f"c{constants_counter}", 1
            )
            constants_counter += 1
        return equation

    def beam_search(self, beam_width, max_depth):
        n_blocks = len(self.building_blocks)
        B = np.identity(n_blocks, dtype=np.uint8)
        I = np.identity(n_blocks, dtype=np.uint8)

        for depth in range(min(n_blocks, max_depth)):
            if B.shape[0] == 0:
                break

            losses_and_configs = self._get_losses_and_configs_parallel(B)
            losses = [loss for loss, _ in losses_and_configs] 

            relevant_losses = np.argsort(losses)[:beam_width]
            self.logger.info(f"Best loss at depth {depth}: {losses[relevant_losses[0]]}")

            beam_matrixes = []
            for current_loss in relevant_losses:
                current_loss_row = B[current_loss, :]
                current_loss_indexes = np.where(current_loss_row == 1)[0]

                # create a matrix of the current loss row
                current_loss_matrix = np.array([current_loss_row]*I.shape[0]).reshape(I.shape)

                # Multiply with the identity matrix
                current_loss_matrix = current_loss_matrix | I

                # Remove the current loss rows
                current_loss_matrix_clean = np.delete(current_loss_matrix, current_loss_indexes, 0)

                # Add the current beam matrix
                beam_matrixes.append(current_loss_matrix_clean)

            B = np.vstack(beam_matrixes)
            B = np.unique(B, axis=0)
        

    

if __name__ == "__main__":
    # Testing branch 
    logging.basicConfig(level=logging.INFO)

    # Create a sythetic dataset with features x1, x2, x3, x4 and target y
    x1 = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
    x2 = np.linspace(0, 10, 1000) + np.random.normal(0, 0.1, 1000)
    x3 = np.random.normal(0, 1, 1000).cumsum() + np.random.normal(0, 0.1, 1000)
    x4 = (np.sin(np.linspace(0, 50, 1000)) + np.random.normal(0, 0.1, 1000)) * 2

    y = 2 * x1 + 0.5 * x2 - 0.3 * x3 + np.sin(x4)

    data = pd.DataFrame({
        'x0': x1,
        'x1': x2,
        'x2': x3,
        'x3': x4,
        'target': y
    })

    edts = EDTS(
        data=data,
        feature_names=["x0", "x1", "x2", "x3"],
        building_blocks=[
            "c_*x_",
            "c_*x_*x_",
            # "c_*x_**2",
            "c_*sin(x_)"
        ],
        iterations=1000,
    )

    edts.beam_search(beam_width=10000, max_depth=4)

    print(edts.best_equation)