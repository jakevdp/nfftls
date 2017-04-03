import numpy as np
from ..utils import complex_exponential_sum, simple_complex_exponential_sum

import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize('yshape', [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize('method', ['auto', 'slow', 'ndft', 'nfft'])
@pytest.mark.parametrize('Nf', [50, 100])
@pytest.mark.parametrize('Nt', [50, 100])
def test_simple_complex_exponential_sum(yshape, method, Nf, Nt, rseed=42):
    rand = np.random.RandomState(rseed)
    t = -0.5 + np.sort(rand.rand(Nt))
    y = rand.randn(*(yshape + (Nt,)))

    baseline = simple_complex_exponential_sum(t, y, Nf, method='slow')
    result = simple_complex_exponential_sum(t, y, Nf, method=method)

    assert result.shape == yshape + (2 * Nf,)
    assert_allclose(result, baseline)


@pytest.mark.parametrize('Nt', [50, 100])
@pytest.mark.parametrize('yshape', [(), (1,), (2,), (1, 2)])
@pytest.mark.parametrize('method', ['auto', 'direct', 'slow', 'ndft', 'nfft'])
@pytest.mark.parametrize('Nf', [50, 100])
@pytest.mark.parametrize('df', [0.01, 0.02])
@pytest.mark.parametrize('f0', [-0.1, 0, 1])
def test_complex_exponential_sum(yshape, Nt, f0, df, Nf, method, rseed=42):
    rand = np.random.RandomState(rseed)
    t = 10 * np.sort(rand.rand(Nt))
    y = rand.randn(*(yshape + (Nt,)))

    baseline = complex_exponential_sum(t, y, f0=f0, Nf=Nf, df=df, method='direct')
    result = complex_exponential_sum(t, y, f0=f0, Nf=Nf, df=df, method=method)

    assert result.shape == yshape + (Nf,)
    assert_allclose(result, baseline)
