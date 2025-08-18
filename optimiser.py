import re
from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp
from sympy.utilities.lambdify import lambdify

class Optimiser:
    def __init__(
            self,
            var_values: Dict[str, jnp.ndarray],
            y: jnp.ndarray,
            lr: float = 1e-2,
            loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
        ):
        self.var_values = var_values
        self.y = y
        self.lr = lr
        self.loss_fn = loss_fn if loss_fn is not None else self._loss_fn

    @staticmethod
    def _loss_fn(pred, y):
        pred = jnp.asarray(pred)
        y = jnp.asarray(y)
        return jnp.mean((pred - y) ** 2)

    def _extract_constants_and_variables(
        self, equation: sp.Expr
    ) -> Tuple[List[sp.Symbol], List[sp.Symbol]]:
        """Split free symbols into constants c0,c1,... and ordinary variables."""
        consts, vars_ = [], []
        for s in equation.free_symbols:
            if re.fullmatch(r"c\d+", s.name):
                consts.append(s)
            else:
                vars_.append(s)

        # Sort constants by their integer suffix (c0, c1, ...)
        consts.sort(key=lambda s: int(s.name[1:]))
        # Sort variables by name for deterministic arg order
        vars_.sort(key=lambda s: s.name)
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

        def model_fn(params: jnp.ndarray, var_values: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            # Gather variable arrays in the same order as `variables`
            var_args = [var_values[name] for name in var_names]
            # Call the lambdified function with *params then *var_args
            return f(*list(params), *var_args)

        return model_fn

    def optimise(
        self,
        equation: sp.Expr,
        iterations: int,
        *,
        loss_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
        init_params: Optional[jnp.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Perform gradient descent (with JAX) to fit constants c0,c1,... in `equation`.

        Args:
            equation: SymPy expression using constants named c0, c1, ...
            iterations: number of gradient steps
            loss_fn: optional custom loss: (pred, y) -> scalar
            init_params: optional initial parameter vector (len = #constants)
            verbose: print loss every ~10% of iterations if True

        Requires:
            self.var_values: dict mapping non-constant symbol names -> JAX arrays
            self.y: target JAX array
            (optional) self.lr: learning rate (default 1e-2)

        Returns:
            dict mapping "c0","c1",... -> optimized float values
        """
        # 1) Identify constants and variables
        constants, variables = self._extract_constants_and_variables(equation)
        if len(constants) == 0:
            raise ValueError("No optimisable constants found (expected symbols c0, c1, ...).")

        # 2) Build model function
        model_fn = self._build_model_fn(equation, constants, variables)

        # 3) Prepare data
        if not hasattr(self, "var_values"):
            raise AttributeError("self.var_values not found. Provide variable arrays for the equation variables.")
        if not hasattr(self, "y"):
            raise AttributeError("self.y not found. Provide target array as self.y.")

        # Sanity-check variable coverage
        missing = [v.name for v in variables if v.name not in self.var_values]
        if missing:
            raise ValueError(f"Missing variable values for: {missing}")

        # 5) Init params
        if init_params is None:
            key = jax.random.PRNGKey(np.random.randint(0, 1_000_000))
            init_params = jax.random.normal(key, (len(constants),), dtype=jnp.float32)
        else:
            init_params = jnp.asarray(init_params)
            if init_params.shape != (len(constants),):
                raise ValueError(f"init_params must have shape {(len(constants),)}")

        lr = getattr(self, "lr", 1e-2)

        # 6) Define objective(param_vector) -> scalar
        def objective(params: jnp.ndarray) -> jnp.ndarray:
            pred = model_fn(params, self.var_values)
            return self.loss_fn(pred, self.y)

        # 7) JIT + grad
        grad_objective = jax.jit(jax.grad(objective))
        objective_jit = jax.jit(objective)

        # 8) SGD loop
        params = init_params
        best_loss = jnp.inf
        for it in range(int(iterations)):
            g = grad_objective(params)
            params = params - lr * g

            current_loss = objective_jit(params)
            if current_loss < best_loss:
                best_loss = current_loss

            if verbose and (it % max(1, iterations // 10) == 0 or it == iterations - 1):
                print(f"[{it:5d}] loss={float(current_loss):.6f}  ||g||={float(jnp.linalg.norm(g)):.6f}")

        # 9) Return as a {name: value} dict (host Python floats)
        result = {c.name: float(params[i]) for i, c in enumerate(constants)}
        return best_loss, result

if __name__ == "__main__":
    x = jnp.linspace(0, 2*jnp.pi, 1000)
    true_c0, true_c1 = 1.7, -0.25
    y = (true_c0 * x) + true_c1 + 0.05 * jax.random.normal(jax.random.PRNGKey(0), (x.size,))

    # Build sympy equation
    x_sym = sp.Symbol("x")
    c0, c1 = sp.symbols("c0 c1")
    equation = (c0 * x_sym) + c1

    opt = Optimiser(
        var_values={"x": x},
        y=y,
        lr=5e-2,
    )
    result = opt.optimise(equation, iterations=1000, verbose=True)

    print(result)  # ~ {'c0': 1.7..., 'c1': -0.25...}
