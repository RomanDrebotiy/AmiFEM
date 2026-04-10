from __future__ import annotations
from typing import Callable
import numpy as np
from enum import Enum

from finite_elements.base_element import eps


class FuncType(Enum):
    TRIAL = 1
    TEST = 2
    COMPOSITE = 3
    TERMINAL = 4


class Expression:
    def __init__(
            self,
            eval_func: Callable[[float, float], np.ndarray] | None = None,
            func_type: FuncType = FuncType.COMPOSITE,
            trial: Expression | None = None,
            test: Expression | None = None
    ):
        self._eval_func: Callable[[float, float], np.ndarray] | None = eval_func
        self.func_type: FuncType = func_type
        self.trial: Expression | None = trial
        self.test: Expression | None = test

    def eval(self, x: float, y: float) -> np.ndarray:
        if self._eval_func is None:
            raise NotImplementedError()
        return self._eval_func(x, y)

    def set_eval(self, eval_func: Callable[[float, float], np.ndarray] | None):
        self._eval_func = eval_func

    def eval_trial_test_pair(self, trial_func: Callable[[float, float], np.ndarray], test_func: Callable[[float, float], np.ndarray], x: float, y: float, clear: bool = False):
        if self.trial is not None:
            self.trial.set_eval(trial_func)
        if self.test is not None:
            self.test.set_eval(test_func)
        res = self.eval(x, y)
        if clear:
            self.clear_trial_test_pair()
        return res

    def clear_trial_test_pair(self):
        if self.trial is not None:
            self.trial.set_eval(None)
        if self.test is not None:
            self.test.set_eval(None)

    def __get_trial_test_pair(self):
        return (
            self if self.func_type == FuncType.TRIAL else self.trial,
            self if self.func_type == FuncType.TEST else self.test
        )

    def __get_trial_test_pair_other(self, other):
        return (
            other if other.func_type == FuncType.TRIAL else other.trial,
            other if other.func_type == FuncType.TEST else other.test
        )

    def __op_aux(self, other, matrix_op: Callable[[np.ndarray, np.ndarray], np.ndarray], exclude_numbers: bool) -> Expression:
        if not exclude_numbers and isinstance(other, (int, float)):
            eval_func=lambda x, y, ev=self.eval, other=other: ev(x, y) * other
            trial, test = self.__get_trial_test_pair()
        elif isinstance(other, np.ndarray):
            eval_func=lambda x, y, matrix_op=matrix_op, ev=self.eval, other=other: matrix_op(ev(x, y), other)
            trial, test = self.__get_trial_test_pair()
        elif isinstance(other, Expression):
            eval_func=lambda x, y, matrix_op=matrix_op, ev=self.eval, other=other: matrix_op(ev(x, y), other.eval(x, y))
            self_trial, self_test = self.__get_trial_test_pair()
            other_trial, other_test = self.__get_trial_test_pair_other(other)
            trial = self_trial or other_trial
            test = self_test or other_test
        else:
            raise NotImplementedError()
        return Expression(
            eval_func=eval_func,
            func_type=FuncType.COMPOSITE,
            trial=trial,
            test=test
        )

    def matr_mul(self, a, b):
        if a.shape == (1,):
            return a[0] * b
        if b.shape == (1,):
            return b[0] * a
        return a @ b

    def __mul__(self, other):
        return self.__op_aux(other=other, matrix_op=lambda a, b: self.matr_mul(a, b), exclude_numbers=False)

    def __rmul__(self, other):
        return self.__op_aux(other=other, matrix_op=lambda a, b: self.matr_mul(b, a), exclude_numbers=False)

    def __add__(self, other):
        return self.__op_aux(other=other, matrix_op=lambda a, b: a + b, exclude_numbers=True)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return other + (-1) * self


class TrialFunction(Expression):
    def __init__(self):
        super().__init__(None, FuncType.TRIAL)


class TestFunction(Expression):
    def __init__(self):
        super().__init__(None, FuncType.TEST)


class Function(Expression):
    def __init__(self, func: Callable[[float, float], np.ndarray]):
        super().__init__(func, FuncType.TERMINAL)


def derivative_directional(expr: Expression, dx: float, dy: float) -> Expression:
    return Expression(
        eval_func=lambda x, y, expr=expr, dx=dx, dy=dy: ((expr.eval(x + dx, y + dy) - expr.eval(x, y)) / np.linalg.norm([dx, dy])),
        func_type=FuncType.COMPOSITE,
        trial=expr if expr.func_type == FuncType.TRIAL else expr.trial,
        test=expr if expr.func_type == FuncType.TEST else expr.test
    )


def der_x(expr: Expression) -> Expression:
    return derivative_directional(expr, eps, 0)


def der_y(expr: Expression) -> Expression:
    return derivative_directional(expr, 0, eps)


def grad(expr: Expression) -> Expression:
    return Expression(
        eval_func=lambda x, y, expr=expr: np.array([der_x(expr).eval(x, y).flatten()[0], der_y(expr).eval(x, y).flatten()[0]]),
        func_type=FuncType.COMPOSITE,
        trial=expr if expr.func_type == FuncType.TRIAL else expr.trial,
        test=expr if expr.func_type == FuncType.TEST else expr.test
    )


def div(expr: Expression) -> Expression:
    return Expression(
        eval_func=lambda x, y: np.array([der_x(expr).eval(x, y).flatten()[0] + der_y(expr).eval(x, y).flatten()[1]]),
        func_type=FuncType.COMPOSITE,
        trial=expr if expr.func_type == FuncType.TRIAL else expr.trial,
        test=expr if expr.func_type == FuncType.TEST else expr.test
    )