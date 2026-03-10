import numpy as np
import matplotlib.pyplot as plt
import pytest
from fdtd1d import FDTD1D

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0)/sigma)**2)

C = 1.0

def test_example():
    # Given...
    num1 = 1
    num2 = 1

    # When...
    result = num1 + num2

    # Expect...
    assert result == 2

def test_fdtd_solves_one_wave():
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05
    initial_e = gaussian(x, x0, sigma)
    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 0.2
    fdtd.run_until(t_final)

    e_solved = fdtd.get_e()

    e_expected = 0.5 * gaussian(x, -t_final*C, sigma) \
     + 0.5 * gaussian(x, t_final*C, sigma)
    
    assert np.allclose(e_solved, e_expected)

if __name__ == "__main__":
    pytest.main([__file__])
