import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fdtd1d import FDTD1D

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

def main():
    C = 1.0

    # ---------------- SIMULATION SETUP ----------------
    x = np.linspace(-1, 1, 201)
    dx = x[1] - x[0]
    x_h = 0.5 * (x[:-1] + x[1:])

    sigma = 0.05
    initial_e = gaussian(x, 0, sigma)

    fdtd = FDTD1D(x)
    fdtd.load_initial_field(initial_e)
    dt = dx / C
    t = 0.0

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    line_e, = ax.plot(x, fdtd.get_e(), label="E field")
    line_h, = ax.plot(x_h, np.zeros_like(x_h), label="H field")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("Field amplitude")
    ax.set_title("1D FDTD Electromagnetic Wave")
    ax.legend()
    ax.grid()

    # ---------------- ANIMATION ----------------
    def update(frame):
        nonlocal t
        t += dt
        fdtd.run_until(t)
        e = fdtd.get_e()
        h = (e[1:] - e[:-1]) / dx  # approximate H
        line_e.set_ydata(e)
        line_h.set_ydata(h)
        return line_e, line_h

    ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
