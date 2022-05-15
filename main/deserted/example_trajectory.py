import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data

# Lorenz System
Lorenz = True
if Lorenz:
    Start_pos = list(np.random.rand(3))
    Time, Trajectory = data.lorenz(length=2000, sample=0.01, x0=Start_pos, discard=0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(Trajectory[:, 0], Trajectory[:, 1], Trajectory[:, 2])

# Henon System
Henon = True
if Henon:
    Start_pos = list(np.random.rand(2))
    Trajectory = data.henon(2000, Start_pos, discard=0)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(Trajectory[:, 0], Trajectory[:, 1], s=1)

# Mackey Glass System
Mackey_glass = True
if Mackey_glass:
    Start_pos = list(np.random.rand(1000))
    Trajectory = data.mackey_glass(2000, Start_pos, discard=1000)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(Trajectory[:-50], Trajectory[50:])

# Roessler System
Roessler = True
if Roessler:
    Start_pos = list(np.random.rand(3))
    Time, Trajectory = data.roessler(2000, Start_pos, discard=0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(Trajectory[:, 0], Trajectory[:, 1], Trajectory[:, 2])
