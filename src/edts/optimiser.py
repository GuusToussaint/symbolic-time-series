"""
EDTS (Equation Discovery for Time Series) is a
framework for discovering equations
"""
import re
from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
from sympy.utilities.lambdify import lambdify


class Optimiser:
    """"
    Optimiser class for fitting symbolic equations to data using gradient
    descent with JAX.
    """

    def __init__(
        self,
        var_values: Dict[str, jnp.ndarray],
        target_values: jnp.ndarray,
        lr: float = 1e-2,
        l2_weight: float = 1e-2,  # Add a regularization weight argument
    ):
        self.var_values = var_values
        self.y = target_values
        self.lr = lr
        self.l2_weight = l2_weight

    @staticmethod
    def loss_fn(pred, target):
        """
        Computes the mean squared error (MSE) loss between predictions and
        targets.

        Args:
            pred: Predicted values, array-like or JAX array.
            y: Ground truth target values, array-like or JAX array.

        Returns:
            jnp.ndarray: The mean squared error between `pred` and `y`.
        """
        pred = jnp.asarray(pred)
        target = jnp.asarray(target)
        return jnp.mean((pred - target) ** 2)

    def _extract_constants_and_variables(
        self, equation: sp.Expr
    ) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """
        Split free symbols into constants c0,c1,... and ordinary variables.
        """
        consts, vars_ = [], []
        for s in equation.free_symbols:
            if re.fullmatch(r"c\d+", str(s)):
                consts.append(s)
            else:
                vars_.append(s)

        # Sort constants by their integer suffix (c0, c1, ...)
        consts.sort(key=lambda s: int(str(s)[1:]))
        # Sort variables by name for deterministic arg order
        vars_.sort(key=str)
        return consts, vars_

    def _build_model_fn(
        self,
        equation: sp.Expr,
        constants: List[sp.Symbol],
        variables: List[sp.Symbol],
    ) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
        """
        Returns model_fn(params, var_values) -> prediction
        where params is a 1D array aligned with `constants`.
        """
        # Create a JAX-compatible callable: args = constants + variables
        f = lambdify(constants + variables, equation, modules="jax")

        var_names = [v.name for v in variables]

        def model_fn(
            params: jnp.ndarray, var_values: Dict[str, jnp.ndarray]
        ) -> jnp.ndarray:
            # Gather variable arrays in the same order as `variables`
            var_args = [var_values[name] for name in var_names]
            # Call the lambdified function with *params then *var_args
            return f(*list(params), *var_args)

        return model_fn

    def optimise(
        self,
        equation: sp.Expr,
        iterations: int,
        init_params: Optional[jnp.ndarray] = None,
        verbose: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Perform gradient descent (with JAX) to fit constants c0,c1,... in
        `equation`.

        Args:
            equation: SymPy expression using constants named c0, c1, ...
            iterations: number of gradient steps
            loss_fn: optional custom loss: (pred, y) -> scalar
            init_params: optional initial parameter vector (len = #constants)
            verbose: print loss every ~10% of iterations if True

        Requires:
            self.var_values: dict mapping non-constant symbol names -> JAX
            arrays
            self.y: target JAX array
            (optional) self.lr: learning rate (default 1e-2)

        Returns:
            dict mapping "c0","c1",... -> optimized float values
        """
        # 1) Identify constants and variables
        constants, variables = self._extract_constants_and_variables(equation)
        if len(constants) == 0:
            raise ValueError(
                "No optimisable constants found (expected symbols c0, c1)"
            )

        # 2) Build model function
        model_fn = self._build_model_fn(equation, constants, variables)

        # 3) Prepare data
        if not hasattr(self, "var_values"):
            raise AttributeError(
                "self.var_values not found. Provide variable arrays for the \
                equation variables."
            )
        if not hasattr(self, "y"):
            raise AttributeError(
                "self.y not found. Provide target array as self.y."
            )

        # Sanity-check variable coverage
        missing = [v.name for v in variables if v.name not in self.var_values]
        if missing:
            raise ValueError(f"Missing variable values for: {missing}")

        # 5) Init params
        if init_params is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
            init_params = jax.random.normal(
                key, (len(constants),), dtype=jnp.float32
            )
        else:
            init_params = jnp.asarray(init_params)
            if init_params.shape != (len(constants),):
                raise ValueError(
                    f"init_params must have shape {(len(constants),)}"
                )

        lr = getattr(self, "lr", 1e-2)

        # 6) Define objective(param_vector) -> scalar
        def objective(params: jnp.ndarray) -> jnp.ndarray:
            pred = model_fn(params, self.var_values)
            mse_loss = self.loss_fn(pred, self.y)
            l2_reg = self.l2_weight * jnp.sum(params ** 2)
            return mse_loss + l2_reg

        # 7) JIT + grad
        grad_objective = jax.jit(jax.grad(objective))
        objective_jit = jax.jit(objective)

        # 8) SGD loop
        params = init_params
        best_loss = jnp.inf
        for it in range(int(iterations)):
            g = grad_objective(params)
            max_norm = 1.0
            grad_norm = jnp.linalg.norm(g)
            if grad_norm > max_norm:
                g = g * (max_norm / grad_norm)
            params = params - lr * g

            print(params)

            current_loss = objective_jit(params)
            if current_loss < best_loss:
                best_loss = current_loss

            if verbose and (
                it % max(1, iterations // 10) == 0 or it == iterations - 1
            ):
                print(
                    f"[{it:5d}] loss={float(current_loss):.6f}  "
                    f"||g||={float(jnp.linalg.norm(g)):.6f}"
                )

        # 9) Return as a {name: value} dict (host Python floats)
        result_dict = {
            c.name: float(params[i])
            for i, c in enumerate(constants)
        }
        return best_loss, result_dict
