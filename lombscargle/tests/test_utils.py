import numpy as np
from ..utils import complex_exponential_sum, simple_complex_exponential_sum

import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def data(rseed=42, N=100):
    rand = np.random.RandomState(rseed)
    t = -0.5 + np.sort(rand.rand(N))
    y = rand.randn(N) + 1j * rand.randn(N)
    return t, y


@pytest.mark.parametrize('method', ['auto', 'slow', 'ndft', 'nfft'])
@pytest.mark.parametrize('Nf', [20, 50, 100])
def test_simple_complex_exponential_sum(data, method, Nf):
    t, y = data
    assert_allclose(simple_complex_exponential_sum(t, y, Nf, method='slow'),
                    simple_complex_exponential_sum(t, y, Nf, method=method))


@pytest.mark.parametrize('method', ['auto', 'direct', 'slow', 'ndft', 'nfft'])
@pytest.mark.parametrize('Nf', [20, 50, 100])
@pytest.mark.parametrize('df', [0.01, 0.02])
@pytest.mark.parametrize('f0', [-0.1, 0, 1])
def test_complex_exponential_sum(data, f0, df, Nf, method):
    t, y = data
    t = 10 * (0.5 + t)
    assert_allclose(complex_exponential_sum(t, y, f0=f0, Nf=Nf, df=df, method='direct'),
                    complex_exponential_sum(t, y, f0=f0, Nf=Nf, df=df, method=method))
