from lal import MSUN_SI, G_SI, C_SI
import numpy as np
from bilby.gw.eos import eos, tov_solver
from scipy.interpolate import interp1d


def tov_integrate(central_pressure, gammas, p0=1.64e33, e0=2.05e14, xmax=6.73):

    geom_to_SI_pressure = C_SI ** 4. / G_SI
    geom_to_SI_mass = C_SI ** 2. / G_SI

    spectral = eos.SpectralDecompositionEOS(gammas, p0=p0, e0=e0, xmax=xmax)
    energy_density = spectral.energy_from_pressure(
        central_pressure / geom_to_SI_pressure, interp_type='linear')

    if hasattr(energy_density, '__len__'):
        sol = np.array([tov_solver.IntegrateTOV(
            spectral, eps, interp_type='linear').integrate_TOV()
                        for eps in energy_density])
        mass, radius, k2 = sol.T
    else:
        mass, radius, k2 = tov_solver.IntegrateTOV(
            spectral, energy_density, interp_type='linear').integrate_TOV()

    return mass * geom_to_SI_mass, radius, k2


def tov_quantities_of_mass(mass, central_pressure, gammas,
                           maximum_mass_lower_limit=None, return_radius=True,
                           return_k2=True, integration_result=None):

    if integration_result is None:
        mass_tmp, radius_tmp, k2_tmp = tov_integrate(central_pressure, gammas)
    else:
        integration_result_ = np.copy(integration_result)
        mass_tmp, radius_tmp, k2_tmp = integration_result_
    mass_tmp /= MSUN_SI

    if hasattr(central_pressure, '__len__'):
        if maximum_mass_lower_limit is not None:
            arg_maximum_mass = np.argmax(mass_tmp)
            max_mass = mass_tmp[arg_maximum_mass]

            if max_mass >= maximum_mass_lower_limit:
                mass_tmp = mass_tmp[:arg_maximum_mass]
                radius_tmp = radius_tmp[:arg_maximum_mass]
                k2_tmp = k2_tmp[:arg_maximum_mass]
            else:
                return np.nan, np.nan

    if hasattr(mass, '__len__'):
        mass_shape = np.shape(mass)
        mass = np.array(mass).flatten()

    if return_radius:
        radius_interp = interp1d(
            mass_tmp, radius_tmp, fill_value=np.nan, bounds_error=False)
        radius = radius_interp(mass)
        if hasattr(mass, '__len__'):
            radius[radius < 0] = 0
            radius = np.reshape(radius, mass_shape)
    else:
        radius = np.nan

    if return_k2:
        k2_interp = interp1d(
            mass_tmp, k2_tmp, fill_value=np.nan, bounds_error=False)
        k2 = k2_interp(mass)
        if hasattr(mass, '__len__'):
            k2[k2 < 0] = 0
            k2 = np.reshape(k2, mass_shape)
    else:
        k2 = np.nan

    return radius, k2


def compactness_of_mass(mass, central_pressure, gammas,
                        maximum_mass_lower_limit=None):

    radius = tov_quantities_of_mass(
        mass, central_pressure, gammas,
        maximum_mass_lower_limit, return_k2=False)[0]

    return G_SI * mass * MSUN_SI / (C_SI**2 * radius)


def second_love_number_of_mass(mass, central_pressure, gammas,
                               maximum_mass_lower_limit=None):

    return tov_quantities_of_mass(
        mass, central_pressure, gammas,
        maximum_mass_lower_limit, return_radius=False)[1]


def Lambda_of_mass(mass, central_pressure, gammas,
                   maximum_mass_lower_limit=None, integration_result=None):

    radius, k2 = tov_quantities_of_mass(
        mass, central_pressure, gammas,
        maximum_mass_lower_limit, integration_result=integration_result)

    return (2/3) * k2 * (C_SI**2 * radius / (G_SI * mass * MSUN_SI))**5


def energy_density_of_pressure(pressure, gammas, p0=1.64e33, e0=2.05e14,
                               xmax=6.73):
    '''
    Parameters
    ----------
    pressure_array = numpy array
        pressure array in SI units
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        polytrope hyper parameters in SI units

    Returns
    -------
    density in kg/m^3
    '''

    geom_to_SI = C_SI ** 4. / G_SI
    spectral = eos.SpectralDecompositionEOS(gammas, p0=p0, e0=e0, xmax=xmax)
    energy_density = spectral.energy_from_pressure(
        pressure / geom_to_SI, interp_type='linear')

    return energy_density * geom_to_SI


def pressure_of_energy_density(energy_density, gammas, p0=1.64e33, e0=2.05e14,
                               xmax=6.73):
    pressure_tmp = np.logspace(np.log10(1e31), np.log10(1e36), 100)
    energy_density_tmp = energy_density_of_pressure(
        pressure_tmp, gammas, p0, e0, xmax)

    return np.interp(
        energy_density, energy_density_tmp, pressure_tmp, left=np.nan,
        right=np.nan)


def maximum_speed_of_sound(gammas, p0=1.64e33, e0=2.05e14, xmax=6.73):
    '''
    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3 = spectral decomposition
    hyper parameters

    Returns
    -------
    Maximum speed of sound divided by the speed of light
    '''

    spectral = eos.SpectralDecompositionEOS(gammas, p0=p0, e0=e0, xmax=xmax)

    pmax = spectral.pressure[-1]
    emax = spectral.energy_from_pressure(pmax, interp_type='linear')
    hmax = spectral.pseudo_enthalpy_from_energy_density(
        emax, interp_type='linear')

    return spectral.velocity_from_pseudo_enthalpy(hmax)


def maximum_pressure(gammas, p0=1.64e33, e0=2.05e14, xmax=6.73):
    '''
    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition hyper-parameters in SI units

    Returns
    -------
    returns the maximum pressure in SI units
    '''

    spectral = eos.SpectralDecompositionEOS(gammas, p0=p0, e0=e0, xmax=xmax)

    return spectral.pressure[-1] * C_SI ** 4. / G_SI
