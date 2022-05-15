import numpy as np
import matplotlib.pyplot as plt
import reservoir_computing as rc

Num = 10
Start_pos = np.random.rand(Num) * 2 * np.pi
Omega = np.ones(Num)
Omega[:2] = 0.9
Omega[2:5] = 0.6
Omega[5:] = 0.4

Time, Sol = rc.kuramoto(length=6000, sample=0.01, 
                        x0=Start_pos, 
                        omega=Omega, k=0.5, discard=2000)

Lines = plt.plot(Sol[:, :5], c='r')
Lines = plt.plot(Sol[:, 5:], c='b')
