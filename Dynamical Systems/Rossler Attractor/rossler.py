import numpy as np

def rossler_derivatives(state, t, a=0.2, b=0.2, c=5.7):
    """Oblicza pochodne dla układu Rösslera."""
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

def solve_rossler(initial_state, t_span, a=0.2, b=0.2, c=5.7):
    """Rozwiązuje układ równań za pomocą scipy.integrate.odeint."""
    from scipy.integrate import odeint
    return odeint(rossler_derivatives, initial_state, t_span, args=(a, b, c))