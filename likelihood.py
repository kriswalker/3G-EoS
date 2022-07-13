import numpy as np
import bilby
import toast.piecewise_polytrope as piecewise_polytrope
from scipy.special import logsumexp


class HyperLikelihood(bilby.Likelihood):
    """
    An hyperlikelihood that samples equation of state hyper-parameters
    assuming the piecewise polytrpe parametrisation.
    See Eq. (9) in https://arxiv.org/pdf/1909.02698.pdf

    Parameters
    ----------
    likelihood_list: list
        A list of interpolated likelihoods
    min_max_values_list: list
        A list containing the minimum and maximum values used for training
        the likelihood distribution
    number_of_grid_points: int
        Number of grid points used for integration
    pressure_array: numpy array
        Pressure array in SI units
    maximum_mass: float
        maxium mass of the EoS determined by pulsar observations in units of
        solar mass, usually 1.97 M_sun
    maximum_speed_of_sound: float
        maximum speed of sound/speed of light (dimensionless)


    Returns
    -------
    Likelihood: `bilby.core.likelihood.Likelihood`
        A bilby likelihood object
    """

    def __init__(self, likelihood_list, logz, min_max_values_list,
                 number_of_grid_points, pressure_array, maximum_mass,
                 maximum_speed_of_sound, log_p=33, Gamma_1=2, Gamma_2=2,
                 Gamma_3=2):
        parameters = dict(log_p=log_p, Gamma_1=Gamma_1, Gamma_2=Gamma_2,
                          Gamma_3=Gamma_3)
        bilby.Likelihood.__init__(self, parameters=parameters)

        self.min_max_values_list = min_max_values_list
        self.number_of_grid_points = number_of_grid_points
        self.cm_vals, self.q_vals, self.dA_array, \
            self.l1_vals, self.l2_vals = self.create_grid()
        self.likelihood_list = likelihood_list
        self.pressure_array = pressure_array
        self.maximum_mass = maximum_mass
        self.maximum_speed_of_sound = maximum_speed_of_sound
        self.logz = logz

    def create_grid(self):
        """
        Creates an integration grid between the minimum and maximum values
        allowed by the posterior
        """
        cm_array = list()
        q_array = list()
        dA_array = list()
        l1_array = list()
        l2_array = list()
        for min_max in self.min_max_values_list:
            cm_array_tmp = np.linspace(
                min_max["min_cm"], min_max["max_cm"],
                self.number_of_grid_points
                )
            q_array_tmp = np.linspace(
                min_max["min_q"], min_max["max_q"],
                self.number_of_grid_points
                )

            dA_array.append((cm_array_tmp[1]-cm_array_tmp[0]) *
                            (q_array_tmp[1]-q_array_tmp[0]))

            cm_array.append(cm_array_tmp)
            q_array.append(q_array_tmp)

            l1_array_tmp = np.linspace(
                min_max["min_lambda_1"], min_max["max_lambda_1"],
                self.number_of_grid_points
                )
            l2_array_tmp = np.linspace(
                min_max["min_lambda_2"], min_max["max_lambda_2"],
                self.number_of_grid_points
                )

            l1_array.append(l1_array_tmp)
            l2_array.append(l2_array_tmp)

        return cm_array, q_array, dA_array, l1_array, l2_array

    def eos_model(self, cm, q, log_p, Gamma_1, Gamma_2, Gamma_3,
                  min_max_values):
        """
        Computes the tidal deformability

        Parameters
        ----------
        chirp_mass: numpy array
        mass_ratio: numpy array
        log_p: float
        Gamma_1: float
        Gamma_2: float
        Gamma_3: float

        Returns
        -------
        Lambda: numpy array
            The tidal deformabilites Lambda_1 and Lambda_2
        """
        dict_tmp = dict(chirp_mass=cm,
                        mass_ratio=q)
        tmp = \
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(
                dict_tmp)

        m1 = tmp[0]['mass_1']
        m2 = tmp[0]['mass_2']
        mass_array = np.array([m1, m2])
        Lambda, max_mass = piecewise_polytrope.Lambda_of_mass(
            self.pressure_array, mass_array, log_p,
            Gamma_1, Gamma_2, Gamma_3, maximum_mass_limit=self.maximum_mass)

        if type(Lambda) == float:
            return None, None
        else:
            return Lambda[0], Lambda[1]

    def get_marg_likelihood(self, likelihood, cm, q, lambda1, lambda2, event):

        m_likelihood = likelihood[0]
        l_likelihood = likelihood[1]

        marg_l = []
        for i, cm_i in enumerate(cm):
            marg_l_i = []
            if (min(self.l1_vals[event]) <= lambda1[i] <= max(
                    self.l1_vals[event])):
                l1_ind = np.argmin(np.abs(lambda1 - self.l1_vals[event]))
                for j, q_i in enumerate(q):
                    if (min(self.l2_vals[event]) <= lambda2[j] <= max(
                            self.l2_vals[event])):
                        l2_ind = np.argmin(np.abs(lambda2[j] -
                                                  self.l2_vals[event]))
                        marg_l_i.append(m_likelihood[i, j] +
                                        l_likelihood[l1_ind, l2_ind])
                    else:
                        marg_l_i.append(-200)
            else:
                marg_l_i = -200 * np.ones(np.shape(cm))
            marg_l.append(marg_l_i)

        return np.array(marg_l)

    def mass_prior(self, cm, q_arr):

        return np.ones(np.shape(q_arr))

    def integrate_likelihood(self, interpolated_likelihood, log_p, Gamma_1,
                             Gamma_2, Gamma_3, cm, q, min_max_values, event):
        """
        Function to be integrated

        Parameters
        ----------
        chirp_mass: numpy array
        mass_ratio: numpy array
        interpolated_likelihood: sklearn.ensemble.RandomForestRegressor
        min_max_values: dict
        log_p: float
        Gamma_1: float
        Gamma_2: float
        Gamma_3: float

        Returns
        -------
        numpy array containing the integrand

        """

        l1, l2 = self.eos_model(cm, q, log_p, Gamma_1, Gamma_2, Gamma_3,
                                min_max_values)

        if l1 is not None:
            marg_likelihood = self.get_marg_likelihood(interpolated_likelihood,
                                                       cm, q, l1, l2, event)
            marg_likelihood = marg_likelihood + self.logz[event]
            # marg_likelihood += np.log(self.mass_prior(cm, q))
            return logsumexp(logsumexp(marg_likelihood, axis=1), axis=0)
        else:
            return -np.inf

    def log_likelihood(self):
        max_speed_of_sound = piecewise_polytrope.maximum_speed_of_sound(
            self.parameters['log_p'], self.parameters['Gamma_1'],
            self.parameters['Gamma_2'], self.parameters['Gamma_3'])

        if max_speed_of_sound <= self.maximum_speed_of_sound:
            integral = list()
            for ii, likelihood in enumerate(self.likelihood_list):
                integral.append(self.integrate_likelihood(
                    likelihood,
                    self.parameters['log_p'],
                    self.parameters['Gamma_1'],
                    self.parameters['Gamma_2'],
                    self.parameters['Gamma_3'],
                    self.cm_vals[ii],
                    self.q_vals[ii],
                    self.min_max_values_list[ii],
                    ii))
            log_l = sum(np.array(integral) + np.log(self.dA_array))
        else:
            log_l = -np.inf

        return np.double(log_l)
