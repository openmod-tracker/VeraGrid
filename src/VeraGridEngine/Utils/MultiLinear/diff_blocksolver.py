# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
import time
import numpy as np
import math
import uuid
import warnings
import scipy.sparse as sp
from typing import Optional

from VeraGridEngine.Devices.Dynamic.events import RmsEvents
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver
from VeraGridEngine.Utils.Symbolic.symbolic import Var, Const, Expr, Func, cos, sin, _emit
from VeraGridEngine.Utils.MultiLinear.differential_var import DiffVar, LagVar
from enum import Enum
from scipy.sparse import csc_matrix
import numba as nb
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Dict, Mapping, Union, List, Sequence, Tuple, Set, Literal
from scipy.sparse.linalg import gmres, spilu, LinearOperator, MatrixRankWarning
from scipy.linalg import LinAlgError, LinAlgWarning
from scipy.sparse.linalg._dsolve import MatrixRankWarning
from VeraGridEngine.Utils.Sparse.csc import pack_4_by_4_scipy

# from VeraGridEngine.Utils.Symbolic.events import EventParam

NUMBER = Union[int, float]
NAME = 'name'


# -----------------------------------------------------------------------------
# UUID helper
# -----------------------------------------------------------------------------

def _new_uid() -> int:
    """Generate a fresh UUID‑v4 string."""
    return uuid.uuid4().int

def _compile_equations(eqs: Sequence[Expr],
                       uid2sym_vars: Dict[int, str],
                       uid2sym_params: Dict[int, str],
                       add_doc_string: bool = True) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:

    """
    Compile the array of expressions to a function that returns an array of values for those expressions
    :param eqs: Iterable of expressions (Expr)
    :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
    :param uid2sym_params:
    :param add_doc_string: add the docstring?
    :return: Function pointer that returns an array
    """
    # Build source
    src = f"def _f(vars, params):\n"
    src += f"    out = np.zeros({len(eqs)})\n"
    src += "\n".join([f"    out[{i}] = {_emit(e, uid2sym_vars, uid2sym_params)}" for i, e in enumerate(eqs)]) + "\n"
    src += f"    return out"
    ns: Dict[str, Any] = {"math": math, "np": np}
    exec(src, ns)
    fn = nb.njit(ns["_f"], fastmath=True)

    if add_doc_string:
        fn.__doc__ = "def _f(vars)"
    return fn

@dataclass(frozen=False)
class DiffBlock(Block):
    diff_vars: List[DiffVar] = field(default_factory=list)
    lag_vars: List[DiffVar] = field(default_factory=list)
    reformulated_vars: List[DiffVar] = field(default_factory=list)
    differential_eqs: List[Expr] = field(default_factory=list)

class DiffBlockSolver(BlockSolver):
    differential_vars : List[DiffVar]

    def __init__(self, block_system: Block):
        """
        Constructor        
        :param block_system: BlockSystem
        """
        self.block_system: Block = block_system

        # Flatten the block lists, preserving declaration order
        self._algebraic_vars: List[Var] = list()
        self._algebraic_eqs: List[Expr] = list()
        self._algebraic_eqs_substituted: List[Expr] = list()
        self._state_vars: List[Var] = list()
        self._state_eqs: List[Expr] = list()
        self._state_eqs_substituted: List[Expr] = list()
        self._diff_vars: List[DiffVar] = list()
        self._differential_eqs: List[Expr] = list()
        self._lag_vars: List[LagVar] = list()
        self._lag_vars_set: Set[LagVar] = set()
        self._reformulated_vars: List[Var] = list()
        self._parameters: List[Var] = list()

        for b in self.block_system.get_all_blocks():
            self._algebraic_vars.extend(b.algebraic_vars)
            self._algebraic_eqs.extend(b.algebraic_eqs)
            self._state_vars.extend(b.state_vars)
            self._state_eqs.extend(b.state_eqs)
            self._parameters.extend(b.parameters)

            if isinstance(b, DiffBlock):
                self._diff_vars.extend(b.diff_vars)
                self._lag_vars.extend(b.lag_vars)
                self._differential_eqs.extend(b.differential_eqs)
                self._reformulated_vars.extend(b.reformulated_vars)

        self._lag_vars_set = set(self._lag_vars)
        self._state_eqs_substituted = self._state_eqs.copy()

        #We define the parameter dt
        self.dt = Var(name='dt')
        self._parameters.append(self.dt)

        self._n_state = len(self._state_vars)
        self._n_alg = len(self._algebraic_vars)
        self._n_vars = self._n_state + self._n_alg
        self._n_params = len(self._parameters)
        self._n_diff = len(self._diff_vars)

        # generate the in-code names for each variable
        # inside the compiled functions the variables are
        # going to be represented by an array called vars[]

        uid2sym_vars: Dict[int, str] = dict()
        uid2sym_params: Dict[int, str] = dict()
        uid2sym_diff: Dict[int, str] = dict()
        self.uid2idx_vars: Dict[int, int] = dict()
        self.uid2idx_params: Dict[int, int] = dict()
        self.uid2idx_diff: Dict[int, int] = dict()
        self.uid2idx_lag: Dict[int, int] = dict()
        i = 0
        for v in self._state_vars:
            uid2sym_vars[v.uid] = f"vars[{i}]"
            self.uid2idx_vars[v.uid] = i
            i += 1

        for v in self._algebraic_vars:
            uid2sym_vars[v.uid] = f"vars[{i}]"
            self.uid2idx_vars[v.uid] = i
            i += 1

        j = 0
        for j, ep in enumerate(self._parameters):
            uid2sym_params[ep.uid] = f"params[{j}]"
            self.uid2idx_params[ep.uid] = j
            j += 1

        k = 0
        for ep in self._diff_vars:
            uid2sym_diff[ep.uid] = f"diff[{k}]"
            self.uid2idx_diff[ep.uid] = k
            k += 1

        #We substitute the differential variable by the Forward Approximation:
        self.alpha = 1
        alpha = self.alpha
        lag_can_be_0 = True
        if lag_can_be_0:
            lag_init = 0
        else:
            lag_init = 1

        for iter, eq in enumerate(self._state_eqs_substituted):
            for var in self._algebraic_vars + self._state_vars:
                deriv = eq.diff(var)
                if getattr(deriv, 'value', 1) == 0:
                    continue 
                lag_var = LagVar.get_or_create(var.name + '_lag_' + str(1),
                                                 base_var=var, lag = 1)
                approximation = alpha*var + (1-alpha)*lag_var
                self._lag_vars_set.add(lag_var)
                eq = eq.subs({var:approximation})
            self._state_eqs_substituted[iter] = eq

        for iter, eq in enumerate(self._state_eqs_substituted):
            for var in self._diff_vars:
                deriv = eq.diff(var)
                if getattr(deriv, 'value', 1) == 0:
                    continue 
                approximation, total_lag = var.approximation_expr(self.dt, lag_can_be_0=lag_can_be_0)
                eq = eq.subs({var:approximation})
                self._lag_vars_set.update(LagVar.get_or_create( var.origin_var.name+ '_lag_' + str(lag),
                                                 base_var=var.origin_var, lag = lag) for lag in range(lag_init, max(3, total_lag+1)))
            self._state_eqs_substituted[iter] = eq

        n_algebraic = len(self._algebraic_eqs)
        self._algebraic_eqs_substituted = self._algebraic_eqs.copy() + self._differential_eqs.copy()
        for iter, eq in enumerate(self._algebraic_eqs_substituted):
            for var in self._algebraic_vars + self._state_vars:
                deriv = eq.diff(var)
                if getattr(deriv, 'value', 1) == 0:
                    continue 
                if var not in self._reformulated_vars:
                    continue
                lag_var_0 = LagVar.get_or_create(var.name + '_lag_' + str(0),
                                                 base_var=var, lag = 0)
                lag_var = LagVar.get_or_create(var.name + '_lag_' + str(1),
                                                 base_var=var, lag = 1)
                if iter < n_algebraic:
                    approximation = alpha*var + (1-alpha)*lag_var
                else:
                    alpha2 = 0.66
                    approximation = alpha2*var + (1-alpha2)*lag_var
                    approximation = alpha2*var + (1-alpha2)*lag_var
                self._lag_vars_set.add(lag_var)
                self._lag_vars_set.add(lag_var_0)
                eq = eq.subs({var:approximation})
                
            for var in self._diff_vars:
                deriv = eq.diff(var)
                if getattr(deriv, 'value', 1) == 0:
                    continue 
                approximation, total_lag = var.approximation_expr(self.dt, lag_can_be_0=lag_can_be_0, central =False)
                eq = eq.subs({var:approximation})
                self._lag_vars_set.update(LagVar.get_or_create( var.origin_var.name+ '_lag_' + str(lag),
                                                 base_var=var.origin_var, lag = lag) for lag in range(lag_init, max(3, total_lag+1)))

            self._algebraic_eqs_substituted[iter] = eq
            
        i = len(self.uid2idx_vars)
        l = 0
        self._lag_vars = sorted(self._lag_vars_set, key=lambda x: (x.base_var.uid, x.lag))
        for v in self._lag_vars:  # deterministic
            uid2sym_vars[v.uid] = f"vars[{i}]"
            self.uid2idx_vars[v.uid] = i
            self.uid2idx_lag[v.uid] = l
            i += 1
            l += 1

        # Compile RHS and Jacobian

        """
                   state Var   algeb var  
        state eq |J11        | J12       |    | ∆ state var|    | ∆ state eq |
                 |           |           |    |            |    |            |
                 ------------------------- x  |------------|  = |------------|
        algeb eq |J21        | J22       |    | ∆ algeb var|    | ∆ algeb eq |
                 |           |           |    |            |    |            |
        """
        print("Compiling...", end="")
        self._rhs_state_fn = _compile_equations(eqs=self._state_eqs_substituted, uid2sym_vars=uid2sym_vars,
                                                uid2sym_params=uid2sym_params)

        self._rhs_algeb_fn = _compile_equations(eqs=self._algebraic_eqs_substituted, uid2sym_vars=uid2sym_vars,
                                                uid2sym_params=uid2sym_params)

        self._j11_fn = self._get_jacobian(eqs=self._state_eqs_substituted, variables=self._state_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params, dt = self.dt)
        self._j12_fn = self._get_jacobian(eqs=self._state_eqs_substituted, variables=self._algebraic_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params, dt = self.dt)
        self._j21_fn = self._get_jacobian(eqs=self._algebraic_eqs_substituted, variables=self._state_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params, dt = self.dt)
        self._j22_fn = self._get_jacobian(eqs=self._algebraic_eqs_substituted, variables=self._algebraic_vars, uid2sym_vars=uid2sym_vars,
                                     uid2sym_params=uid2sym_params, dt = self.dt)
        _ = 0
    
    def jacobian_implicit(self, x: np.ndarray, params: np.ndarray, h: float) -> sp.csc_matrix:
        """
        :param x: vector or variables' values
        :param params: params array
        :param h: step
        :return:
        """

        """
                  state Var    algeb var
        state eq |I - h * J11 | - h* J12  |    | ∆ state var|    | ∆ state eq |
                 |            |           |    |            |    |            |
                 -------------------------- x  |------------|  = |------------|
        algeb eq |J21         | J22       |    | ∆ algeb var|    | ∆ algeb eq |
                 |            |           |    |            |    |            |
        """
        I = sp.eye(m=self._n_state, n=self._n_state)
        j11: sp.csc_matrix = (I - h * self._j11_fn(x, params)).tocsc()
        j12: sp.csc_matrix = - h * self._j12_fn(x, params)
        j21: sp.csc_matrix = self._j21_fn(x, params)
        j22: sp.csc_matrix = self._j22_fn(x, params)
        J = pack_4_by_4_scipy(j11, j12, j21, j22)
        return J

    def _get_jacobian(self,
                      eqs: List[Expr],
                      variables: List[Var],
                      uid2sym_vars: Dict[int, str],
                      uid2sym_params: Dict[int, str],
                      dt: Const = Const(0.001)):
        """
        JIT‑compile a sparse Jacobian evaluator for *equations* w.r.t *variables*.
        :param eqs: Array of equations
        :param variables: Array of variables to differentiate against
        :param uid2sym_vars: dictionary relating the uid of a var with its array name (i.e. var[0])
        :param uid2sym_params:
        :return:
                jac_fn : callable(values: np.ndarray) -> scipy.sparse.csc_matrix
                    Fast evaluator in which *values* is a 1‑D NumPy vector of length
                    ``len(variables)``.
                sparsity_pattern : tuple(np.ndarray, np.ndarray)
                    Row/col indices of structurally non‑zero entries.
        """

        # Ensure deterministic variable order
        diff_vars = self._diff_vars
        check_set = set()
        for v in variables:
            if v in check_set:
                raise ValueError(f"Repeated var {v.name} in the variables' list :(")
            else:
                check_set.add(v)

        # Cache compiled partials by UID so duplicates are reused
        fn_cache: Dict[str, Callable] = {}
        triplets: List[Tuple[int, int, Callable]] = []  # (col, row, fn)

    
        for row, eq in enumerate(eqs):
            for lag_var in self._lag_vars:
                if lag_var.lag == 0:
                    eq = eq.subs({lag_var: lag_var.base_var})

            for col, var in enumerate(variables):
                d_expression = eq.diff(var).simplify()
                for diff_var in diff_vars:
                    deriv = eq.diff(diff_var)
                    continue
                    if getattr(deriv, 'value', 1) != 0 and diff_var.origin_var.uid == var.uid:
                        name_diff = 'diff_' + diff_var.name
                        diff_2_var = DiffVar.get_or_create(name = name_diff, base_var=diff_var)
                        dx2_dt2, lags1 = diff_2_var.approximation_expr(dt=dt)
                        dx_dt, lags2   = diff_var.approximation_expr(dt=dt)
                        #d_expression = dx_dt*d_expression + dx2_dt2*eq.diff(diff_var).simplify()
                        #To use the expression above we would need to multiply every value by dxdt and change the RHS
                        #TO DO: see if its really needed to ensure multilinearity
                        d_expression += (dx2_dt2/dx_dt)*eq.diff(diff_var).simplify()

                        set_lags = set( LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(lag), 
                                                base_var = diff_var.origin_var, lag = lag) for lag in range(1, max(lags1, lags2)))
                        new_lags = set_lags - self._lag_vars_set

                        #We add the lag to the index
                        self._lag_vars_set.update(new_lags)
                        i = len(self.uid2idx_vars)
                        l = len(self.uid2idx_lag)
                        for v in new_lags:  # deterministic
                            uid2sym_vars[v.uid] = f"vars[{i}]"
                            self.uid2idx_vars[v.uid] = i
                            self.uid2idx_lag[v.uid] = l
                            self._lag_vars.append(v)
                            i += 1
                            l += 1
                        
                        k = len(self.uid2idx_diff)
                        if diff_2_var not in self._diff_vars:
                            self.uid2idx_diff[diff_2_var.uid] = k
                            k += 1

                
                #We substitute the remaining diff vars in d_expression
                for diff_var in diff_vars:
                    deriv = d_expression.diff(diff_var)
                    if getattr(deriv, 'value', 1) != 0:

                        dx_dt, lag = diff_var.approximation_expr(dt=dt, lag_can_be_0=False)
                        d_expression = d_expression.subs({diff_var: dx_dt})
                        new_lag = LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(lag), 
                                                base_var = diff_var.origin_var, lag = lag)
                        i = len(self.uid2idx_vars)
                        l = len(self.uid2idx_lag)
                        if new_lag not in self._lag_vars_set:
                            uid2sym_vars[new_lag.uid] = f"vars[{i}]"
                            self.uid2idx_vars[new_lag.uid] = i
                            self.uid2idx_lag[new_lag.uid] = l
                            self._lag_vars.append(new_lag)
                            self._lag_vars_set.add(new_lag)
                            i += 1
                            l += 1

                if isinstance(d_expression, Const) and d_expression.value == 0:
                    continue  # structural zero

                function_ptr = _compile_equations(eqs=[d_expression], uid2sym_vars=uid2sym_vars,
                                                  uid2sym_params=uid2sym_params)

                fn = fn_cache.setdefault(d_expression.uid, function_ptr)

                triplets.append((col, row, fn))

        # Sort by column, then row for CSC layout
        triplets.sort(key=lambda t: (t[0], t[1]))
        cols_sorted, rows_sorted, fns_sorted = zip(*triplets) if triplets else ([], [], [])

        nnz = len(fns_sorted)
        indices = np.fromiter(rows_sorted, dtype=np.int32, count=nnz)
        data = np.empty(nnz, dtype=np.float64)

        indptr = np.zeros(len(variables) + 1, dtype=np.int32)
        for c in cols_sorted:
            indptr[c + 1] += 1
        np.cumsum(indptr, out=indptr)

        def jac_fn(values: np.ndarray, params) -> sp.csc_matrix:  # noqa: D401 – simple
            assert len(values) >= len(variables)
            for k, fn_ in enumerate(fns_sorted):
                data[k] = fn_(values, params)
            return sp.csc_matrix((data, indices, indptr), shape=(len(eqs), len(variables)))

        return jac_fn
    
    def sort_vars(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros(len(self._state_vars) + len(self._algebraic_vars), dtype=object)

        for key, val in mapping.items():
            i = self.uid2idx_vars[key.uid]
            x[i] = key

        return x

    def build_init_diffvars_vector(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros( len(self._diff_vars) )

        for key, val in mapping.items():
            if key.uid in self.uid2idx_diff.keys():
                i = self.uid2idx_diff[key.uid]
                x[i] = val
            else:
                raise ValueError(f"Missing variable {key} definition")

        return x

    def build_init_lagvars_vector(self, mapping: dict[Var, float]) -> np.ndarray:
        """
        Helper function to build the initial vector
        :param mapping: var->initial value mapping
        :return: array matching with the mapping, matching the solver ordering
        """
        x = np.zeros( len(self._lag_vars) )

        for key, val in mapping.items():
            if key.uid in self.uid2idx_vars.keys():
                try:
                    i = self.uid2idx_vars[key.uid]
                    x[i] = val
                except:
                    _=0
            else:
                raise ValueError(f"Missing variable {key} definition")

        return x

    def build_initial_lag_variables(self, x0:np.ndarray, dx0:np.ndarray, h) -> np.ndarray:
        if len(self._lag_vars) == 0:
            return

        x_lag = np.zeros(len(self._lag_vars), dtype=np.float64)

        lag_registry = self._lag_vars[0]._registry
        diff_registry = self._diff_vars[0]._absolute_registry

        max_order = max(var.diff_order for var in self._diff_vars)
        max_order = max(2, max_order)
        filtered_lag_dict = { key: value for key, value in lag_registry.items() if key[1] <= max_order }
        sorted_lag_dict = sorted(filtered_lag_dict.items(), key=lambda item: (item[0][0], item[0][1]))

        for key, lag_var in sorted_lag_dict:
            base_var_uid, lag = key

            uid = lag_var.uid 

            if base_var_uid not in self.uid2idx_vars or lag_var not in self._lag_vars:
                continue
            idx = self.uid2idx_lag[uid] 
            x0_uid = self.uid2idx_vars[base_var_uid]
            if lag == 0: 
                x_lag[idx] = x0[x0_uid]
                continue
            # Collect previous dx0 and x_lag values for this lag_var
            dx0_slice = np.zeros(lag_var.lag)
            x_lag_last = 0

            for (prev_uid, prev_lag), prev_var in lag_registry.items():
                if prev_uid == base_var_uid and prev_lag <= lag and prev_lag !=0: 
                    try:
                        prev_diff     = diff_registry[base_var_uid, prev_lag]
                        prev_idx_diff = self.uid2idx_diff[prev_diff.uid]
                        dx0_slice[prev_lag-1] = dx0[prev_idx_diff]
                    
                    except:
                        if (base_var_uid, 1) in diff_registry and diff_registry[base_var_uid, 1] in self._diff_vars:
                            prev_diff     = diff_registry[base_var_uid, 1]
                            prev_idx_diff = self.uid2idx_diff[prev_diff.uid]
                            dx0_slice[prev_lag-1] = dx0[prev_idx_diff]
                        else:
                            dx0_slice[prev_lag-1] = 0


                    

            lag_i = lag_var.populate_initial_lag(x0[x0_uid], dx0_slice, x_lag_last, self.dt, h)
            if isinstance(lag_i, Expr):
                x_lag[idx] = lag_i.eval(dt = h)
            else:
                x_lag[idx] = lag_i
            _ = 0
        return x_lag
    
    def build_initial_guess(self, x0:np.ndarray, dx0:np.ndarray, h) -> np.ndarray:
        res = x0.copy() 
        for diff_var in self._diff_vars:
            if diff_var.diff_order > 1:
                continue 
            uid = diff_var.base_var.uid
            idx = self.uid2idx_vars[uid]
            diff_idx = self.uid2idx_diff[diff_var.uid]
            res[idx] += h*dx0[diff_idx]
        return res

    def simulate(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            dx0: np.ndarray,
            params0: np.ndarray,
            events_list: RmsEvents,
            method: Literal["rk4", "euler", "implicit_euler"] = "rk4",
            newton_tol: float = 1e-8,
            newton_max_iter: int = 1000,
            followed_vars = None,
            verbose =False

    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param events_list:
        :param params0:
        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        lag0 = self.build_initial_lag_variables(x0, dx0, h)
        x0   = self.build_initial_guess(x0, dx0, h)
        params_matrix = self.build_params_matrix(int(np.ceil((t_end - t0) / h)), params0, events_list)
        if method == "euler":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="euler")
        if method == "rk4":
            return self._simulate_fixed(t0, t_end, h, x0, params0, stepper="rk4")
        if method == "implicit_euler":

            return self._simulate_implicit_euler(
                t0, t_end, h, x0, dx0, lag0, params0, params_matrix,
                tol=newton_tol, max_iter=newton_max_iter, verbose = verbose, followed_vars = followed_vars
            )
        raise ValueError(f"Unknown method '{method}'")

    def _simulate_implicit_euler(self, t0, t_end, h, x0, dx0, lag0, params0: np.ndarray, diff_params_matrix, tol=1e-8,
                                 max_iter=1000, followed_vars = None, verbose = False):
        """
        :param t0:
        :param t_end:
        :param h:
        :param x0:
        :params_matrix:
        :param tol:
        :param max_iter:
        :return:
        """
        steps = int(np.ceil((t_end - t0) / h))
        t = np.empty(steps + 1)
        y = np.empty((steps + 1, self._n_vars))
        speed_up = 1.0
        self.y = y
        self.t = t 
        params_current = params0
        diff_params_matrix = diff_params_matrix
        t[0] = t0
        y[0] = x0.copy()
        dx = dx0.copy()
        lag = np.asarray(lag0, dtype=np.float64)
        for step_idx in range(steps):
            self.step_idx = step_idx
            params_previous = params_current.copy()
            params_current += diff_params_matrix[step_idx, :].toarray().ravel()
            discontinuity = np.linalg.norm(params_current - params_previous, np.inf) > 1e-10
            xn = y[step_idx]
            x_last = y[step_idx-1] if step_idx > 0 else y[step_idx]
            x_last_lags = np.r_[x_last, lag]
            x_new = xn.copy()  # initial guess
            converged = False
            n_iter = 0
            lambda_reg = 1e-6  # small regularization factor
            max_reg_tries = 1e6  # limit how much regularization is added
            reg_attempts = 0

            #We compute dx for the next step
            dx = self.compute_dx(x_new, lag, h)
            if step_idx == 0:
                tol = 1e-4
            else:
                tol = 1e-8
            while not converged and n_iter < max_iter:
                self._update_0_lags(x_new, lag)
                xn_lags = np.r_[xn, lag]
                xnew_lags = np.r_[x_new, lag]

                if discontinuity:
                    tol = 1e-8
                    max_iter = 5e5
                    #lag = self.build_initial_lag_variables(x_new, dx, h)
                    #print(f'discontinuity at t = {t[step_idx]}, lag reset to {lag}')
                
                if followed_vars is not None:
                    for var in followed_vars:
                        idx = self.get_var_idx(var) 
                        #xnew_lags[idx] = (xnew_lags[idx] + x_last_lags[idx])/2

                params_current = np.asarray(params_current, dtype=np.float64)                    
                rhs = self.rhs_implicit(xnew_lags, xn_lags, params_current, step_idx + 1, h)
                Jf = self.jacobian_implicit(xnew_lags, params_current, h)  # sparse matrix
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", MatrixRankWarning)
                        delta = sp.linalg.spsolve(Jf, -rhs)
                except MatrixRankWarning as w:
                    raise np.linalg.LinAlgError(f"Jacobian is singular: {w}")



                residual = np.linalg.norm(rhs, np.inf)
                converged = residual < tol

                if step_idx == 0:
                    alpha_update = 0.8
                    old_lag = lag
                    lag = self.build_initial_lag_variables(x_new, dx0, h)
                    lag = (1-alpha_update)*old_lag + alpha_update*lag
                    if converged:
                        print("System well initialized.")
                    else:
                        print(f"System bad initialized. DAE resiudal is {residual}.")

                if converged:
                    break
                
                Jf = self.jacobian_implicit(xnew_lags, params_current, h)  # sparse matrix
                solved = False
                
                delta = sp.linalg.spsolve(Jf, -rhs)
                solved = np.all(np.isfinite(delta))

                if not solved:
                    raise ValueError(
                        f"spsolve returned non-finite values (NaN or Inf).\n"
                        f"delta = {delta}\n"
                        f"rhs = {rhs}\n"
                        f"Jacobian shape = {Jf.shape}"
                    )


                if not solved:
                    raise RuntimeError("Failed to solve linear system even with regularization.")

                x_new += delta
                n_iter += 1

                print(f'lag is {lag}')

            if converged:
                if verbose:
                    print(f'converged {converged} and n_iter {n_iter} and iter {n_iter} and rhs is  {rhs}')

                if discontinuity:
                    _=0
                    y[step_idx + 1] = x_new 
                else:
                    y[step_idx + 1] = x_new
                t[step_idx + 1] = t[step_idx] + h

                for i, lag_var in enumerate(self._lag_vars):
                    if lag_var.lag == 0:
                        uid = lag_var.base_var.uid
                        idx = self.uid2idx_vars[uid]
                        lag[i] = x_new[idx] 
                    elif step_idx + 1 - (lag_var.lag-1) >= 0:
                        uid = lag_var.base_var.uid
                        idx = self.uid2idx_vars[uid]
                        lag[i] = y[step_idx + 1 - (lag_var.lag-1), idx]
                    else:
                        lag_name = lag_var.base_var.name + '_lag_' + str(lag_var.lag-1)
                        next_lag_var = LagVar.get_or_create(lag_name, base_var= lag_var.base_var, lag = lag_var.lag-1)
                        uid = next_lag_var.uid
                        idx = self.uid2idx_lag[uid]
                        lag[i] = lag[idx] 
            else:
                print(f"Failed to converge at step {step_idx}")
                print(f'Jacobian is {Jf}')
                break

        return t, y
    
    def _update_0_lags(self, x_new, lag):
        for i, lag_var in enumerate(self._lag_vars):
            if lag_var.lag == 0:
                uid = lag_var.base_var.uid
                idx = self.uid2idx_vars[uid]
                lag[i] = x_new[idx]
    
    def compute_dx(self, x:np.ndarray, lag: np.ndarray, h: float) -> np.ndarray:
        """
        Compute the numerical derivative (dx) for all differential variables 
        using symbolic approximation expressions and lagged variables.
    
        Parameters
        ----------
        y : np.ndarray
            State variable trajectory. `y[-1, :]` corresponds to the most recent 
            values of the system variables.
        lag : np.ndarray
            Array containing lagged values of variables (delayed states).
        h : float
            Time step (dt) used in the approximation.
    
        Returns
        -------
        np.ndarray
            Array with computed derivatives for each differential variable, 
            indexed consistently with `self._diff_vars`.
        """
        res = np.zeros( len(self._diff_vars), dtype=np.float64)
        for diff_var in self._diff_vars:
            uid = diff_var.uid
            idx = self.uid2idx_diff[uid]
            dx_expression, lag_number = diff_var.approximation_expr(self.dt)

            #We substitute the origin variable and dt
            lag_0 = LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(0),   
                                            base_var=diff_var.origin_var, lag = 0)
            subs = {diff_var.origin_var: Const(x[self.uid2idx_vars[diff_var.origin_var.uid]])}
            subs[lag_0] = Const(x[self.uid2idx_vars[diff_var.origin_var.uid]])
            subs[self.dt] = Const(h)

            #We substitute the lag variables
            i = 1
            lag_in_expression = True
            while lag_in_expression or i<=2:
                lag_i = LagVar.get_or_create(diff_var.origin_var.name+ '_lag_' + str(i),   
                                            base_var=diff_var.origin_var, lag = i)
                dx_expression = dx_expression.subs({self.dt:Const(h)})
                deriv = dx_expression.diff(lag_i)
                if getattr(deriv, 'value', 1) == 0:
                    if i>2:
                        lag_in_expression = False
                        i = i+1
                        break
                    i += 1
                    continue
                lag_idx = self.uid2idx_lag[lag_i.uid]

                subs[lag_i] = Const(lag[lag_idx])
                i += 1

            deriv_value = dx_expression.subs(subs)
            deriv_value = deriv_value.eval()
            res[idx] = deriv_value

        return res
    
    def test_equations(
            self,
            t0: float,
            t_end: float,
            h: float,
            x0: np.ndarray,
            dx0: np.ndarray,
            params0: np.ndarray,
            events_list: RmsEvents,
            method: Literal["rk4", "euler", "implicit_euler"] = "rk4",
            newton_tol: float = 1e-8,
            newton_max_iter: int = 1000,

    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param events_list:
        :param params0:
        :param t0: start time
        :param t_end: end time
        :param h: step
        :param x0: initial values
        :param method: method
        :param newton_tol:
        :param newton_max_iter:
        :return: 1D time array, 2D array of simulated variables
        """
        lag0 = self.build_initial_lag_variables(x0, dx0, h)
        x0   = self.build_initial_guess(x0, dx0, h)

        params_matrix = self.build_params_matrix(int(np.ceil((t_end - t0) / h)), params0, events_list)
        params_current = params0
        params_current += params_matrix[0, :].toarray().ravel()
        xn_lags = np.concatenate((x0, lag0))
        print(f' xn is {xn_lags}, params is {params_current}')
        rhs = self.rhs_implicit(xn_lags, xn_lags, params_current, 0, h)
        print(f"rhs is {rhs}")
        Jf = self.jacobian_implicit(xn_lags, params_current, h)  # sparse matrix
        print(f"Jf is {Jf}")

        return
        
        
