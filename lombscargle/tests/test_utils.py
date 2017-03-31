import numpy as np
from ..utils import trig_sum

import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def data(rseed=42, N=100):
    rand = np.random.RandomState(rseed)
    t = -0.5 + np.sort(rand.rand(N))
    y = rand.randn(N) + 1j * rand.randn(N)
    return t, y


@pytest.mark.parametrize('method', ['slow', 'ndft', 'nfft'])
def test_trig_sum(data, method, Nf=100):
    t, y = data
    assert_allclose(trig_sum(t, y, Nf, method='auto'),
                    trig_sum(t, y, Nf, method=method))
