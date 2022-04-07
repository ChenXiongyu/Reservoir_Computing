import warnings
from nolitsa import data, lyapunov
import numpy as np
import matplotlib.pyplot as plt


def poly_fit(x, y, degree, fit="RANSAC"):
  # check if we can use RANSAC
  if fit == "RANSAC":
    try:
      # ignore ImportWarnings in sklearn
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import sklearn.linear_model as sklin
        import sklearn.preprocessing as skpre
    except ImportError:
      warnings.warn(
        "fitting mode 'RANSAC' requires the package sklearn, using"
        + " 'poly' instead",
        RuntimeWarning)
      fit = "poly"

  if fit == "poly":
    return np.polyfit(x, y, degree)
  elif fit == "RANSAC":
    model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
    xdat = np.asarray(x)
    if len(xdat.shape) == 1:
      # interpret 1d-array as list of len(x) samples instead of
      # one sample of length len(x)
      xdat = xdat.reshape(-1, 1)
    polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
    try:
      model.fit(polydat, y)
      coef = model.estimator_.coef_[::-1]
    except ValueError:
      warnings.warn(
        "RANSAC did not reach consensus, "
        + "using numpy's polyfit",
        RuntimeWarning)
      coef = np.polyfit(x, y, degree)
    return coef
  else:
    raise ValueError("invalid fitting mode ({})".format(fit))


dt = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]
x = data.lorenz(length=4000, sample=dt, x0=x0,
                sigma=16.0, beta=4.0, rho=45.92)[1]
plt.plot(range(len(x)), x)
plt.show()

# Choose appropriate Theiler window.
meanperiod = 30
maxt = 250
d = lyapunov.mle(x, maxt=maxt, window=meanperiod)
t = np.arange(maxt) * dt
coefs = poly_fit(t, d, 1)
print('LLE = ', coefs[0])

plt.title('Maximum Lyapunov exponent for the Lorenz system')
plt.xlabel(r'Time $Max_t$')
plt.ylabel(r'Average divergence $\langle d_i(Max_t) \rangle$')
plt.plot(t, d, label='divergence')
plt.plot(t, t * 1.50, '--', label='slope=1.5')
plt.plot(t, coefs[1] + coefs[0] * t, '--', label='RANSAC')
plt.legend()
plt.show()
