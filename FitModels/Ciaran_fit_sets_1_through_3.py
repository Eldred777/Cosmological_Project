"""
Based off Ciaran_Fit_Modules.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
import multiprocessing as multi
import matplotlib as mpl

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
    # make sure plotting directory is made
    if not os.path.exists(f"{project_dir}/FitModels/plots/{filename}"):
        os.mkdir(f"{project_dir}/FitModels/plots/{filename}")

    zs, mu, muerr = read_data(filename)
    # for speed, we are going to take only a subset of the available data
    zs_sliced = zs[0:-1:_STEP]
    mu_sliced = mu[0:-1:_STEP]
    muerr_sliced = muerr[0:-1:_STEP]
    # print(f"Starting analysis of {filename}.")

    # Set up the arrays for the models you want to test, e.g. a range of Omega_m and Omega_Lambda models:
    oms = np.linspace(0.0, 1.0, _N)  # Array of matter densities
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

    chi2_shifted = chi2 - np.amin(chi2)
    chi2_shifted_transposed = np.transpose(chi2_shifted)

    # Now we want to find error bars.
    likelihood = np.exp(
        -chi2_shifted / 2
    )  # convert the chi^2 to a likelihood (np.amin(chi2) calculates the minimum of the chi^2 array)

    ### find error bars
    # quantiles
    lower_sd_bound = 0.5 - 0.314
    upper_sd_bound = 0.5 + 0.314

    # distribution for omega_l
    likelihood_l = np.sum(likelihood, 0)

    lik_l_cumsum_norm = np.cumsum(likelihood_l) / sum(
        likelihood_l
    )  # should look like a pmf

    # initialise
    sd_lower_l = 0
    sd_upper_l = 0
    for i, x in enumerate(lik_l_cumsum_norm):
        if x > lower_sd_bound and not sd_lower_l:
            # if sd_lower_l not yet found
            sd_lower_l = ols[i]

        if x > upper_sd_bound:
            sd_upper_l = ols[i]
            break  # end loop

    # plot likelihood
    fig, ax = plt.subplots()
    ax.plot(ols, likelihood_l)
    ax.set_xlabel("$\Omega_\Lambda$")
    ax.set_ylabel("Likelihood")
    ax.axvline(ols[ibest[1]], color="red", linestyle=":")
    ax.axvline(sd_lower_l, color=(0, 0, 0, 0.5), linestyle=":")
    ax.axvline(sd_upper_l, color=(0, 0, 0, 0.5), linestyle=":")
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/ol_likelihood.png",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/ol_likelihood.pdf",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )

    plt.close(fig)

    # distribution for omega_m
    likelihood_m = np.sum(likelihood, 1)
    lik_m_cumsum_norm = np.cumsum(likelihood_m) / sum(
        likelihood_m
    )  # should look like a pmf

    # initialise
    sd_lower_m = 0
    sd_upper_m = 0
    for i, x in enumerate(lik_m_cumsum_norm):
        if x > lower_sd_bound and not sd_lower_m:
            # if sd_lower_l not yet found
            sd_lower_m = oms[i]

        if x > upper_sd_bound:
            sd_upper_m = oms[i]
            break  # end loop

    # plot likelihood
    fig, ax = plt.subplots()
    ax.plot(oms, likelihood_m)
    ax.set_xlabel("$\Omega_m$")
    ax.set_ylabel("Likelihood")
    ax.axvline(oms[ibest[0]], color="red", linestyle=":")
    ax.axvline(sd_lower_m, color=(0, 0, 0, 0.5), linestyle=":")
    ax.axvline(sd_upper_m, color=(0, 0, 0, 0.5), linestyle=":")
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/om_likelihood.png",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/om_likelihood.pdf",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )

    plt.close(fig)

    ### plotting time
    # Plot contours of 1, 2, and 3 sigma
    fig, ax = plt.subplots()

    ax.contour(
        oms,
        ols,
        chi2_shifted_transposed,
        cmap="winter",
        **{"levels": [2.30, 6.18, 11.83]},  # corr. 1, 2, 3 sigma
        # ? double check above
    )
    ax.plot(
        oms[ibest[0]],
        ols[ibest[1]],
        "x",
        color="black",
        label=f"($\Omega_m,\Omega_\Lambda$)=({oms[ibest[0]]:.3f},{ols[ibest[1]]:.3f})",
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

    # save 2 versions, one without indicators for standard deviation, and one with
    # without std dev lines
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/contours_sans_sd.png",
        bbox_inches="tight",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/contours_sans_sd.pdf",
        bbox_inches="tight",
    )

    # indicate standard deviations
    # color kwarg gives opacity
    ax.axhline(ols[ibest[1]], color="red", linestyle=":")
    ax.axvline(oms[ibest[0]], color="red", linestyle=":")
    ax.axvline(sd_lower_m, color=(0, 0, 0, 0.5), linestyle=":")
    ax.axvline(sd_upper_m, color=(0, 0, 0, 0.5), linestyle=":")
    ax.axhline(sd_lower_l, color=(0, 0, 0, 0.5), linestyle=":")
    ax.axhline(sd_upper_l, color=(0, 0, 0, 0.5), linestyle=":")

    # with std dev lines
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/contours.png",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/contours.pdf",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )

    plt.close(fig)

    # let's try our hands at making a corner plot!
    fig = plt.figure()
    gs = fig.add_gridspec(
        2, 2, hspace=0.05, wspace=0.05, width_ratios=[1, 0.5], height_ratios=[0.5, 1]
    )
    (om_ax, redundant_ax), (main_ax, ol_ax) = gs.subplots(sharex="col", sharey="row")

    main_ax.contour(
        oms,
        ols,
        chi2_shifted_transposed,
        cmap="winter",
        **{"levels": [2.30, 6.18, 11.83]},  # corr. 1, 2, 3 sigma
    )
    main_ax.plot(  # indicate the parameters of best fit
        oms[ibest[0]],
        ols[ibest[1]],
        "x",
        color="black",
        label=f"($\Omega_m,\Omega_\Lambda$)=({oms[ibest[0]]:.3f},{ols[ibest[1]]:.3f})",
    )
    main_ax.set_xlabel("$\Omega_m$", fontsize=12)
    main_ax.set_ylabel("$\Omega_\Lambda$", fontsize=12)
    main_ax.plot(
        [oms[0], oms[1]],
        [ols[0], ols[1]],
        "-",
        color="black",
        label="Step size indicator",
    )
    main_ax.legend(frameon=False)
    ol_ax.axhline(ols[ibest[1]], color="red", linestyle=":")
    om_ax.axvline(oms[ibest[0]], color="red", linestyle=":")
    main_ax.axvline(sd_lower_m, color=(0, 0, 0, 0.5), linestyle=":")
    main_ax.axvline(sd_upper_m, color=(0, 0, 0, 0.5), linestyle=":")
    main_ax.axhline(sd_lower_l, color=(0, 0, 0, 0.5), linestyle=":")
    main_ax.axhline(sd_upper_l, color=(0, 0, 0, 0.5), linestyle=":")
    main_ax.grid()

    ol_ax.plot(likelihood_l, ols)
    ol_ax.set_xlabel("Likelihood")
    ol_ax.axhline(ols[ibest[1]], color="red", linestyle=":")
    ol_ax.axhline(sd_lower_l, color=(0, 0, 0, 0.5), linestyle=":")
    ol_ax.axhline(sd_upper_l, color=(0, 0, 0, 0.5), linestyle=":")
    ol_ax.grid()

    om_ax.plot(oms, likelihood_m)
    om_ax.set_ylabel("Likelihood")
    om_ax.axvline(oms[ibest[0]], color="red", linestyle=":")
    om_ax.axvline(sd_lower_m, color=(0, 0, 0, 0.5), linestyle=":")
    om_ax.axvline(sd_upper_m, color=(0, 0, 0, 0.5), linestyle=":")
    om_ax.grid()

    redundant_ax.axis("off")

    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/corner.png",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )
    fig.savefig(
        f"{project_dir}/FitModels/plots/{filename}/corner.pdf",
        bbox_inches="tight",
        transparent=False,
        facecolor="w",
    )

    # save all values to a file for later reference
    with open(
        f"{project_dir}/FitModels/plots/{filename}/parameters.txt", "w"
    ) as writer:
        writer.write(
            f"{filename}:"
            + f"\n\t(om,ol)=({oms[ibest[0]]},{ols[ibest[1]]})"
            + f"\n\tReduced chi^2 = {chi2_reduced[ibest[0],ibest[1]]}"
            + f"\n\t{sd_lower_l=}"
            + f"\n\t{sd_upper_l=}"
            + f"\n\t{sd_lower_m=}"
            + f"\n\t{sd_upper_m=}"
        )

    plt.close(fig)


def test_main():
    # when testing, reduce grid search to 20 increments
    global _N
    _N = 20
    analyse_data("Data1")


def main():
    # Analyses data 1-3 simultaneously
    procs = []
    filenames = ["Data0", "Data1", "Data2", "Data3"]

    for filename in filenames:
        procs.append(
            multi.Process(
                target=analyse_data, args=[filename], name=f"Process-{filename}"
            )
        )

    proc: multi.Process
    for proc in procs:
        proc.start()
        print(f"Starting process {proc}")

    for proc in procs:
        proc.join()  # wait until processes are done


if __name__ == "__main__":
    # test_main()
    main()
