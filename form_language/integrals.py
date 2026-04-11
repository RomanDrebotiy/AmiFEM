from __future__ import annotations
from enum import Enum
from typing import Dict, Callable

import numpy as np

from form_language.expressions import Expression


class Measure(Enum):
    DX = 1
    DS = 2


class Integral:
    def __init__(self, measure_map: Dict[Measure, Expression] | None = None):
        self.measure_map: Dict[Measure, Expression] = measure_map
        self.quad_map: Dict[Measure, Callable] = {
            Measure.DX: self.__quad_space,
            Measure.DS: self.__quad_boundary
        }

    def __add__(self, other: Integral):
        res = {**self.measure_map}
        for k, v in other.measure_map.items():
            if k in res:
                res[k] += v
            else:
                res[k] = v
        return Integral(measure_map=res)

    def __sub__(self, other: Integral):
        res = {**self.measure_map}
        for k, v in other.measure_map.items():
            if k in res:
                res[k] -= v
        return Integral(measure_map=res)

    def __quad_space(self, f, v0, v1, v2, v0_on_bnd: bool, v1_on_bnd: bool, v2_on_bnd: bool):
        points = [
            (1 / 2, 1 / 2, 0),
            (1 / 2, 0, 1 / 2),
            (0, 1 / 2, 1 / 2)
        ]
        area = 0.5 * abs(np.cross(v1 - v0, v2 - v0))
        result = 0.0
        for l1, l2, l3 in points:
            p = l1 * v0 + l2 * v1 + l3 * v2
            result += f(p[0], p[1])
        return result * area / 3.0

    def __quad_boundary(self, f, v0, v1, v2, v0_on_bnd: bool, v1_on_bnd: bool, v2_on_bnd: bool):
        vl = [v0, v1, v2, v0]
        vbndl = [v0_on_bnd, v1_on_bnd, v2_on_bnd, v0_on_bnd]
        edges = []
        for i in range(len(vl)-1):
            if vbndl[i] and vbndl[i+1]:
                edges.append([vl[i], vl[i+1]])
        res = 0
        for b, e in edges:
            res += (f(b[0], b[1]) + f(e[0], e[1])) * np.linalg.norm(e - b) / 2.0
        return res

    def eval(self, v0, v1, v2, v0_on_bnd, v1_on_bnd, v2_on_bnd, trial_func: Callable[[float, float], np.ndarray] | None, test_func: Callable[[float, float], np.ndarray] | None) -> float:
        res = 0
        for measure, expr in self.measure_map.items():
            quad = self.quad_map[measure]
            if expr.trial is not None:
                expr.trial.set_eval(trial_func)
            if expr.test is not None:
                expr.test.set_eval(test_func)
            f = lambda x, y: expr.eval(x, y)
            res += quad(f, v0, v1, v2, v0_on_bnd, v1_on_bnd, v2_on_bnd)
            expr.clear_trial_test_pair()
        return res


def integrate(expr: Expression, measure: Measure) -> Integral:
    return Integral(measure_map={measure: expr})
