import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import unyt
from sys import argv

def get_mass_enclosed(df):

    rad = df['radius'].to_numpy(dtype=np.float64)
    dens = df['density'].to_numpy(dtype=np.float64)

    delr = (rad[1] - rad[0])/2.
    inner_r = rad - delr
    outer_r = rad + delr

    # calc mass
    mass_shells =  4. / 3. * np.pi * ( (outer_r)**3 - (inner_r)**3 ) * dens * unyt.g


    mass_arr = mass_shells.cumsum() 
    print(mass_arr[-1].in_units('Msun'))
    return mass_arr.in_units('Msun').value


def plot_vs_x(df, x_var='radius'):
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(4, 2, figure=fig)

    # composition half size
    axCO = fig.add_subplot(gs[0, 0])
    axUrca = fig.add_subplot(gs[1, 0])

    # rest full size
    axRho = fig.add_subplot(gs[2:, 0])
    axP = fig.add_subplot(gs[2:, 1])
    axT = fig.add_subplot(gs[:2, 1])

    # plot composition
    df.plot(x_var, ['carbon-12', 'oxygen-16'], ax=axCO)
    axCO.set_ylabel("Mass fraction")
    axCO.set_title("C-O Mass Fraction")
    axCO.set_xlabel('')
    axCO.grid()

    df.plot(x_var, ['neon-23', 'sodium-23'], ax=axUrca)
    axUrca.set_ylabel("Mass fraction")
    axUrca.set_title("URCA-23 Mass Fraction")
    axUrca.set_xlabel('')
    axUrca.grid()

    # plot temperature
    df.plot(x_var, 'temperature', ax=axT)
    axT.set_yscale('log')
    axT.set_ylabel(" T ")
    axT.set_xlabel('')
    axT.set_title("Temperature")
    axT.grid()

    # plot density
    df.plot(x_var, 'density', ax=axRho)
    axRho.set_yscale('log')
    axRho.set_ylabel("rho (g/cm^3)")
    axRho.set_title("Density")
    axRho.grid()

    # plot pressure
    df.plot(x_var, 'pressure', ax=axP)
    axP.set_yscale('log')
    axP.set_ylabel("pressure")
    axP.set_title("Pressure")
    axP.grid()

    fig.tight_layout(w_pad=1.8)

    return fig


def main(model_file):
    myheader = ['radius', 'density', 'temperature',
                "pressure", "neutron", "hydrogen-1", "helium-4",
                "carbon-12", "oxygen-16", "neon-20", "neon-23",
                "sodium-23", "magnesium-23"]
    # read in data 
    model_df = pd.read_csv(model_file,header=None, skiprows=14,
                           delim_whitespace=True)
    model_df.columns=myheader

    # calc mass enclosed
    mass_enc = get_mass_enclosed(model_df)
    model_df.insert(1, 'mass', mass_enc)

    # plot all the things in rad
    fig_rad = plot_vs_x(model_df, x_var='radius')
    fig_mass = plot_vs_x(model_df, x_var='mass')

    # save
    fig_rad.savefig(f"plots_{model_file}_radius.png")
    fig_mass.savefig(f"plots_{model_file}_mass.png")


if __name__ == "__main__":
    fname = argv[1]
    main(fname)
