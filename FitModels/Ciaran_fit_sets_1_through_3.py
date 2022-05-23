"""
Based off Ciaran_Fit_Modules.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
import multiprocessing as multi

project_dir = os.path.abspath("~/../")

# set some constants
c = 299792.458  # km/s (speed of light)
H0kmsmpc = 70.0  # Hubble constant in km/s/Mpc
H0s = (
    H0kmsmpc * 3.2408e-20
)  # H0 in inverse seconds is H0 in km/s/Mpc * (3.2408e-20 Mpc/km)

# change the following two lines to change how fine the grid search is, and how many data points to take
_STEP = 1  # step size along data
_N = 100  # how many points to consider along the grid


print(f"Check that this path is correct! {project_dir=}")


def ezinv(z, om=0.3, ol=0.7, w0=-1.0, wa=0.0, orr=0.0):
    # integrand, i.e. $1/E(z)$,
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


def Sk(xx, ok):
    # curvature correction function
    if ok < 0.0:
        dk = np.sin(np.sqrt(-ok) * xx) / np.sqrt(-ok)
    elif ok > 0.0:
        dk = np.sinh(np.sqrt(ok) * xx) / np.sqrt(ok)
    else:
        dk = xx
    return dk


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
    d = np.genfromtxt(f"{project_dir}/data/{model_name}.txt", delimiter=",")
    zs = d[:, 0]
    mu = d[:, 1]
    muerr = d[:, 2]
    return zs, mu, muerr


def analyse_data(filename):
    zs, mu, muerr = read_data(filename)
    # for speed, we are going to take only a subset of the available data
    zs_sliced = zs[0:-1:_STEP]
    mu_sliced = mu[0:-1:_STEP]
    muerr_sliced = muerr[0:-1:_STEP]
    # print(f"Starting analysis of {filename}.")

    # Set up the arrays for the models you want to test, e.g. a range of Omega_m and Omega_Lambda models:
    oms = np.linspace(0.0, 0.6, _N)  # Array of matter densities
    ols = np.linspace(0.0, 1.0, _N)  # Array of cosmological constant values
    chi2 = (
        np.ones((_N, _N)) * np.inf
    )  # Array to hold our chi2 values, set initially to super large values

    # Calculate Chi2 for each model
    for i, om in enumerate(oms):  # loop through matter densities
        for j, ol in enumerate(ols):  # loop through cosmological constant densities
            mu_model = dist_mod(
                zs_sliced, om=om, ol=ol
            )  # calculate the distance modulus vs redshift for that model
            mscr = np.sum((mu_model - mu_sliced) / muerr_sliced**2) / np.sum(
                1.0 / muerr_sliced**2
            )  # Calculate the vertical offset to apply
            mu_model_norm = mu_model - mscr  # Apply the vertical offset
            chi2[i, j] = np.sum(
                (mu_model_norm - mu_sliced) ** 2 / muerr_sliced**2
            )  # Calculate the chi2 and save it in a matrix

    # Convert that to a likelihood and calculate the reduced chi2
    # likelihood = np.exp(
    #     -0.5 * (chi2 - np.amin(chi2))
    # )  # convert the chi^2 to a likelihood
    # np.amin(chi2) calculates the minimum of the chi^2 array

    dof = 2
    chi2_reduced = chi2 / (
        len(mu_sliced) - dof
    )  # calculate the reduced chi^2, i.e. chi^2 per degree of freedom, where dof = number of data points minus number of parameters being fitted

    # Calculate the best fit values (where chi2 is minimum)
    indbest = np.argmin(
        chi2
    )  # Gives index of best fit but where the indices are just a single number
    ibest = np.unravel_index(
        indbest, [_N, _N]
    )  # Converts the best fit index to the 2d version (i,j)

    chi2_shifted_transposed = np.transpose(chi2 - np.amin(chi2))

    ### plotting time
    # Plot contours of 1, 2, and 3 sigma
    fig, ax = plt.subplots()

    ax.contour(
        oms,
        ols,
        chi2_shifted_transposed,
        cmap="winter",
        **{"levels": [2.30, 6.18, 11.83]},  # corr. 1, 2, 3 sigma 
        #? double check above 
    )
    ax.plot(
        oms[ibest[0]],
        ols[ibest[1]],
        "x",
        color="black",
        label=f"(om,ol)=({oms[ibest[0]]:.3f},{ols[ibest[1]]:.3f})",
    )
    ax.set_xlabel("$\Omega_m$", fontsize=12)
    ax.set_ylabel("$\Omega_\Lambda$", fontsize=12)
    ax.plot(
        [oms[0], oms[1]],
        [ols[0], ols[1]],
        "-",
        color="black",
        label="Step size indicator",
    )
    ax.legend(frameon=False)
    fig.savefig(
        f"{project_dir}/FitModels/plots/ciaran_contours_{filename}.png",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/ciaran_contours_{filename}.pdf",
        bbox_inches="tight",
    )

    with open(
        f"{project_dir}/FitModels/plots/parameters_{filename}.txt", "w"
    ) as writer:
        print(f"Best fit values are (om,ol)=({oms[ibest[0]]:.3f},{ols[ibest[1]]:.3f})")
        print(
            f"Reduced chi^2 for the best fit is {chi2_reduced[ibest[0],ibest[1]]:0.2f}"
        )
        writer.write(
            f"{filename}:"
            + f"\n\t(om,ol)=({oms[ibest[0]]},{ols[ibest[1]]})"
            + f"\n\tReduced chi^2 = {chi2_reduced[ibest[0],ibest[1]]}"
            + 2 * "\n"
        )

    plt.close(fig)


def main():
    procs = []
    filenames = ["Data1", "Data2", "Data3"]

    for filename in filenames:
        procs.append(multi.Process(target=analyse_data, args=[filename]))

    proc: multi.Process
    for proc in procs:
        proc.start()
        print(f"Starting process {proc}")

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
