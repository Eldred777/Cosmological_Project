"""
Based off Ciaran_Fit_Modules.ipynb.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import corner

# First let's set up our packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import os

cd = os.path.abspath("~/../")

# And set some constants
c = 299792.458  # km/s (speed of light)
H0kmsmpc = 70.0  # Hubble constant in km/s/Mpc
H0s = (
    H0kmsmpc * 3.2408e-20
)  # H0 in inverse seconds is H0 in km/s/Mpc * (3.2408e-20 Mpc/km)

print(f"Check that this path is correct! {cd=}")

# Write a function for the integrand, i.e. $1/E(z)$,
def ezinv(z, om=0.3, ol=0.7, w0=-1.0, wa=0.0, orr=0.0):
    ok = 1.0 - om - ol - orr
    a = 1 / (1 + z)
    wl = w0 + wa * (1 - a)
    ez = np.sqrt(
        orr * (1 + z) ** 4
        + om * (1 + z) ** 3
        + ok * (1 + z) ** 2
        + ol * (1 + z) ** (3 * (1 + wl))
    )
    return 1.0 / ez


# The curvature correction function
def Sk(xx, ok):
    if ok < 0.0:
        dk = np.sin(np.sqrt(-ok) * xx) / np.sqrt(-ok)
    elif ok > 0.0:
        dk = np.sinh(np.sqrt(ok) * xx) / np.sqrt(ok)
    else:
        dk = xx
    return dk


# The distance modulus
def dist_mod(zs, om=0.3, ol=0.7, w0=-1.0, wa=0.0, orr=0.0):
    """Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ol
    xx = np.array(
        [integrate.quad(ezinv, 0, z, args=(om, ol, w0, wa, orr))[0] for z in zs]
    )
    D = Sk(xx, ok)
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)  # Distance modulus
    # Add an arbitrary constant that's approximately the log of c on Hubble constant minus absolute magnitude of -19.5
    dist_mod = (
        dist_mod + np.log(c / H0kmsmpc) - (-19.5)
    )  # You can actually skip this step and it won't make a difference to our fitting
    return dist_mod


# Add a new function that reads in the data (data files should be in a directory called data)
def read_data(model_name):
    d = np.genfromtxt(f"{cd}/data/{model_name}.txt", delimiter=",")
    zs = d[:, 0]
    mu = d[:, 1]
    muerr = d[:, 2]
    return zs, mu, muerr


# for filename in ["Data4", "Data5", "Data6", "Data7"]:
for filename in ["Data4"]:
    zs, mu, muerr = read_data(filename)
    # for speed, we are going to take only a subset of the available data
    step = 10  # step size along zs array
    zs_sliced = zs[0:-1:step]
    mu_sliced = mu[0:-1:step]
    muerr_sliced = muerr[0:-1:step]
    print(f"{filename}: {len(zs)=}")

    # Set up the arrays for the models you want to test, e.g. a range of Omega_m and Omega_Lambda models:
    n = 5
    oms = np.linspace(0.0, 1, n)
    orrs = np.linspace(0.0, 1, n)
    ols = np.linspace(0.0, 1.0, n)
    w0s = np.linspace(-2.0, 0, n)
    was = np.linspace(-1.5, 0.5, n)

    # Array to hold our chi2 values, set initially to super large values
    chi2 = np.ones((n, n, n, n, n)) * np.inf

    # Calculate Chi2 for each model
    for i, om in enumerate(oms):
        for j, ol in enumerate(ols):
            for k, w0 in enumerate(w0s):
                for l, wa in enumerate(was):
                    for m, orr in enumerate(orrs):
                        # calculate the distance modulus vs redshift for that model
                        mu_model = dist_mod(
                            zs_sliced, om=om, ol=ol, w0=w0, wa=wa, orr=orr
                        )
                        # Calculate the vertical offset to apply
                        mscr = np.sum(
                            (mu_model - mu_sliced) / muerr_sliced**2
                        ) / np.sum(1.0 / muerr**2)
                        mu_model_norm = mu_model - mscr  # Apply the vertical offset
                        # Calculate the chi2 and save it in a matrix
                        chi2[i, j, k, l, m] = np.sum(
                            (mu_model_norm - mu_sliced) ** 2 / muerr_sliced**2
                        )

    # # Convert that to a likelihood and calculate the reduced chi2
    # likelihood = np.exp(
    #     -0.5 * (chi2 - np.amin(chi2))
    # )  # convert the chi^2 to a likelihood (np.amin(chi2) calculates the minimum of the chi^2 array)
    chi2_reduced = chi2 / (
        len(mu) - 5
    )  # calculate the reduced chi^2, i.e. chi^2 per degree of freedom, where dof = number of data points minus number of parameters being fitted

    # Calculate the best fit values (where chi2 is minimum)
    ind_best = np.argmin(
        chi2
    )  # Gives index of best fit but where the indices are just a single number
    ibest = np.unravel_index(
        ind_best, [n, n, n, n, n]
    )  # Converts the best fit index to the 2d version (i,j)
    print(f"Index of best fit, {ind_best=} corresponding to {chi2[ibest]=}")
    print(f"Best fit values are (om,ol,)=({oms[ibest[0]]:.3f},{ols[ibest[1]]:.3f})")
    print(
        f"Reduced chi^2 for the best fit is {chi2_reduced[ibest[0],ibest[1],ibest[2],ibest[3], ibest[4]]:0.2f}"
    )

    # samples = np.vstack([oms, orrs, w0s, was, orrs])

    # figure = corner.corner(samples)
    figure = corner.corner(chi2)
    figure.savefig(f"{cd}/plots/{filename}.pdf")
    #! i give up, it is not working properly

    # Plot contours of 1, 2, and 3 sigma
    # fig, ax = plt.subplots()

    # ax.contour(  # ! not working yet
    #     oms,
    #     ols,
    #     np.transpose(chi2 - np.amin(chi2)),
    #     cmap="winter",
    #     **{"levels": [2.30, 6.18, 11.83]},
    # )
    # ax.plot(
    #     oms[ibest[0]],
    #     ols[ibest[1]],
    #     "x",
    #     color="black",
    #     label=f"(om,ol,w0, wa, orr)=({oms[ibest[0]]:.3f},{oms[ibest[1]]:.3f},{oms[ibest[2]]:.3f},{oms[ibest[3]]:.3f},{oms[ibest[4]]:.3f})",
    # )
    # ax.set_xlabel("$\Omega_m$", fontsize=12)
    # ax.set_ylabel("$\Omega_\Lambda$", fontsize=12)
    # ax.plot(
    #     [oms[0], oms[1]],
    #     [ols[0], ols[1]],
    #     "-",
    #     color="black",
    #     label="Step size indicator",
    # )  # Delete this line after making step size smaller!
    # ax.legend(frameon=False)
    # fig.savefig(  # save png
    #     f"{cd}/FitModels/plots/C_contours_{filename}.ngf",
    #     bbox_inches="tight",
    #     transparent=True,
    # )
    # fig.savefig(  # save pdf
    #     f"{cd}/FitModels/plots/C_contours_{filename}.pdf",
    #     bbox_inches="tight",
    #     transparent=True,
    # )

# todo: maybe use a corner plot to visualise this?
