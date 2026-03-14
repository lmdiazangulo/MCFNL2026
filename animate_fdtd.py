import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from fdtd1d import FDTD1D, C


def gaussian(x, x0, sigma):
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def run_animation(save_mp4=False, mp4_name="fdtd_sim.mp4"):
    # Spatial grid
    x = np.linspace(-1, 1, 201)
    x0 = 0.0
    sigma = 0.05

    # Instantiate solver and set initial E; H starts at zero
    fdtd = FDTD1D(x)
    fdtd.load_initial_field(gaussian(x, x0, sigma))

    # Choose final time and number of frames
    t_final = 5 # Inicial : 0.4
    n_steps = int(round((t_final - 0.0) / fdtd.dt))

    # Storage for fields
    Es = []
    Hs = []

    # Step and record
    for _ in range(n_steps + 1):
        Es.append(fdtd.e.copy())
        Hs.append(fdtd.h.copy())
        fdtd._step()

    Es = np.array(Es)
    Hs = np.array(Hs)

    # Prepare figure: plot E on grid nodes, H on staggered half-nodes
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    line_e, = ax[0].plot(x, Es[0], color="C0")
    ax[0].set_ylabel("E")
    ax[0].set_ylim(1.1 * Es.min(), 1.1 * Es.max())
    ax[0].grid(True)

    # H lives between x points; construct half-grid for plotting
    x_h = 0.5 * (x[:-1] + x[1:])
    line_h, = ax[1].plot(x_h, Hs[0], color="C1")
    ax[1].set_ylabel("H")
    ax[1].set_ylim(1.1 * Hs.min(), 1.1 * Hs.max())
    ax[1].set_xlabel("x")
    ax[1].grid(True)

    title = fig.suptitle("")

    def update(frame):
        line_e.set_ydata(Es[frame])
        line_h.set_ydata(Hs[frame])
        title.set_text(f"t = {frame * fdtd.dt:.4f}")
        return line_e, line_h

    # Disable blitting for compatibility with multiple axes and the title
    anim = animation.FuncAnimation(fig, update, frames=len(Es), interval=30, blit=False)

    if save_mp4:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='fdtd'), bitrate=1800)
        anim.save(mp4_name, writer=writer)
        print(f"Saved animation to {mp4_name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # By default show interactive animation. To save mp4 set save_mp4=True
    run_animation(save_mp4=False)
