import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, lyapunov
from outer_function import poly_fit

Start_pos = [1, 1, 1]
Time, Trajectory = data.lorenz(length=4000, sample=0.01, x0=Start_pos, discard=0,
                               sigma=10, beta=8/3, rho=28)

# Choose appropriate Theiler window.
Divergence = lyapunov.mle(Trajectory, maxt=250, window=30)
Max_t = np.arange(250) * 0.01
Coef = poly_fit(Max_t, Divergence, 1)
print('LLE = ', Coef[0])

plt.title('Maximum Lyapunov exponent for the Lorenz system')
plt.xlabel(r'Time $Max_t$')
plt.ylabel(r'Average divergence $\langle d_i(Max_t) \rangle$')
plt.plot(Max_t, Divergence, label='divergence')
plt.plot(Max_t, Coef[1] + Coef[0] * Max_t, '--', label='RANSAC')
plt.legend()

Rossler = False
if Rossler:
    x0 = [1, 1, 1]
    t, x = data.roessler(length=3000, x0=x0, sample=0.05, a=0.2, b=0.2, c=5.7)
    d = lyapunov.mle(x, maxt=250, window=30)
    m = np.arange(250) * 0.01
    c = poly_fit(m, d, 1)
    print('LLE = ', c[0])
    plt.plot(m, d, label='divergence')
    plt.plot(m, c[1] + c[0] * m, '--', label='RANSAC')
