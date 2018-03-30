from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_true, assert_greater
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut

from workbench import Workbench
from workbench.utils import gen_data
from workbench.loo_utils import loo_patterns_from_model


def test_loo_patterns_from_model():
    X, y, A = gen_data(noise_scale=0.1)

    # Normal path
    model = Ridge(fit_intercept=False)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = Ridge(fit_intercept=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = Ridge(normalize=True)
    p0 = Workbench(model).fit(X[1:], y[1:]).pattern_normalized_
    p1, _ = next(loo_patterns_from_model(model, X, y))
    assert_allclose(p0, p1)

    # Optimized path for LinearRegression
    model = LinearRegression(fit_intercept=False)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = LinearRegression(fit_intercept=True)
    patterns = loo_patterns_from_model(model, X, y)
    for train, _ in LeaveOneOut().split(X, y):
        w = Workbench(model).fit(X[train], y[train])
        p0, n0 = w.pattern_, w.normalizer_
        p1, n1 = next(patterns)
        assert_allclose(p0, p1)
        assert_allclose(n0, n1)

    model = LinearRegression(normalize=True)
    p0 = Workbench(model).fit(X[1:], y[1:]).pattern_normalized_
    p1, _ = next(loo_patterns_from_model(model, X, y))
    assert_allclose(p0, p1)
