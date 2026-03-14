import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pytest


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def test_example():
    num1 = 1
    num2 = 1
    assert num1 + num2 == 2


C = 1.0


class FDTD1D:

    def __init__(self, x):
        self.x = x
        self.dx = x[1] - x[0]
        self.dt = self.dx / C

        self.E = np.zeros_like(x)
        self.H = np.zeros(len(x) - 1)

    def load_initial_field(self, initial_e):
        self.E = initial_e.copy()

    def step(self):

        # Update H (half grid)
        self.H += (self.dt / self.dx) * (self.E[1:] - self.E[:-1])

        # Update E
        self.E[1:-1] += (self.dt / self.dx) * (self.H[1:] - self.H[:-1])

        # Boundary conditions
        self.E[0] = 0
        self.E[-1] = 0

    @property
    def get_e(self):
        return self.E

    @property
    def get_h(self):
        return self.H


def test_fdtd_solves_one_wave():
    x = np.linspace(-1, 1, 201)
    x0 = 0
    sigma = 0.05

    initial_e = gaussian(x, x0, sigma)

    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)

    t_final = 0.2
    n_steps = int(round(t_final / fdtd.dt))

    for _ in range(n_steps):
        fdtd.step()

    e_solved = fdtd.get_e

    e_expected = (
        0.5 * gaussian(x, -t_final * C, sigma)
        + 0.5 * gaussian(x, t_final * C, sigma)
    )

    assert np.allclose(e_solved, e_expected, atol=1e-2)


# ---------------- ANIMATION ----------------

x = np.linspace(-1, 1, 201)
x_h = 0.5 * (x[:-1] + x[1:])

sigma = 0.05
initial_e = gaussian(x, 0, sigma)

fdtd = FDTD1D(x)
fdtd.load_initial_field(initial_e)

fig, ax = plt.subplots(figsize=(8, 4))

line_e, = ax.plot(x, fdtd.get_e, label="E field")
line_h, = ax.plot(x_h, fdtd.get_h, label="H field")

ax.set_xlim(-1, 1)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x")
ax.set_ylabel("Field amplitude")
ax.set_title("1D FDTD Electromagnetic Wave")
ax.legend()
ax.grid()


def update(frame):

    fdtd.step()

    line_e.set_ydata(fdtd.get_e)
    line_h.set_ydata(fdtd.get_h)

    return line_e, line_h


ani = FuncAnimation(
    fig,
    update,
    frames=300,
    interval=30,
    blit=True
)

plt.show()