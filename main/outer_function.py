import warnings
import numpy as np
import sklearn.linear_model as sk_lin
import sklearn.preprocessing as sk_pre


def poly_fit(x, y, degree, fit="RANSAC"):
    # check if we can use RANSAC
    # if fit == "RANSAC":
    #     try:
    #         # ignore ImportWarnings in sklearn
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore", ImportWarning)
    #             import sklearn.linear_model as sk_lin
    #             import sklearn.preprocessing as sk_pre
    #     except ImportError:
    #         warnings.warn(
    #             "fitting mode 'RANSAC' requires the package sklearn, using"
    #             + " 'poly' instead",
    #             RuntimeWarning)
    #         fit = "poly"

    if fit == "poly":
        return np.polyfit(x, y, degree)
    elif fit == "RANSAC":
        model = sk_lin.RANSACRegressor(sk_lin.LinearRegression(fit_intercept=False))
        xdat = np.asarray(x)
        if len(xdat.shape) == 1:
            # interpret 1d-array as list of len(x) samples instead of
            # one sample of length len(x)
            xdat = xdat.reshape(-1, 1)
        poly_dat = sk_pre.PolynomialFeatures(degree).fit_transform(xdat)
        try:
            model.fit(poly_dat, y)
            coef = model.estimator_.coef_[::-1]
        except ValueError:
            warnings.warn(
                "RANSAC did not reach consensus, "
                + "using numpy polyfit",
                RuntimeWarning)
            coef = np.polyfit(x, y, degree)
        return coef
    else:
        raise ValueError("invalid fitting mode ({})".format(fit))
