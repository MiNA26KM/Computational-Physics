import pytest
import numpy as np
from rossler import rossler_derivatives

def test_rossler_derivatives_at_origin():
    """Sprawdza wartości pochodnych w punkcie (0,0,0)."""
    # Dla a=0.2, b=0.2, c=5.7 i punktu (0,0,0):
    # dx/dt = -0 - 0 = 0
    # dy/dt = 0 + 0.2*0 = 0
    # dz/dt = 0.2 + 0*(0-5.7) = 0.2
    state = [0, 0, 0]
    t = 0
    derivatives = rossler_derivatives(state, t, a=0.2, b=0.2, c=5.7)
    
    assert derivatives == [0, 0, 0.2]

def test_rossler_stability():
    """Sprawdza czy wynik jest tablicą o odpowiednim kształcie."""
    from rossler import solve_rossler
    t = np.linspace(0, 10, 100)
    result = solve_rossler([1, 1, 1], t)
    assert result.shape == (100, 3)