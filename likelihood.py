import numpy as np
from scipy.special import logsumexp


class EOSLikelihood:
    """
    A hyperlikelihood that samples equation of state hyper-parameters given a
    chosen parametrization. Takes in likelihood as a function of the component
    masses and tidal deformabilities.

    """

    def __init__(self, likelihood_list, axes_list, log_evidence_list,
                 pressure_array, maximum_mass, maximum_speed_of_sound,
                 mass_prior, parametrization, likelihood_floor=-np.inf):

        self.axes_list = axes_list
        self.likelihood_list = likelihood_list
        self.log_evidence_list = log_evidence_list
        self.pressure_array = pressure_array
        self.maximum_mass = maximum_mass
        self.maximum_speed_of_sound = maximum_speed_of_sound
        self.mass_prior = mass_prior
        self.parametrization = parametrization
        self.likelihood_floor = likelihood_floor

        mass_array_list = []
        dA_list = []
        for axes in axes_list:
            m1, m2 = axes[0], axes[2]
            dA = np.diff(m1)[0] * np.diff(m2)[0]
            mass_array_list.append(np.array([m1, m2]))
            dA_list.append(dA)
        self.mass_array_list = mass_array_list
        self.dA_list = dA_list

    def eos_model(self, mass_array, tov_result):
        """

        """

        try:
            Lambda = self.parametrization.Lambda_of_mass(
                mass_array, self.pressure_array, None,
                maximum_mass_lower_limit=self.maximum_mass,
                integration_result=tov_result)
        except:
            return None, None

        return Lambda[0], Lambda[1]

    def get_marg_likelihood(self, likelihood, axes, lambda1, lambda2):

        m1_ax, l1_ax, m2_ax, l2_ax = axes

        cond1 = (max(lambda1) < min(l1_ax)) or (min(lambda1) > max(l1_ax))
        cond2 = (max(lambda2) < min(l2_ax)) or (min(lambda2) > max(l2_ax))
        if cond1 or cond2:
            marg_l = len(m1_ax) * len(m2_ax) * [self.likelihood_floor]
        else:
            marg_l = []
            for i, m1 in enumerate(m1_ax):
                if min(l1_ax) <= lambda1[i] <= max(l1_ax):
                    l1_ind = np.argmin(np.abs(lambda1[i] - l1_ax))
                    for j, m2 in enumerate(m2_ax):
                        if min(l2_ax) <= lambda2[j] <= max(l2_ax):
                            l2_ind = np.argmin(np.abs(lambda2[j] - l2_ax))
                            ml = likelihood[i, l1_ind, j, l2_ind] + \
                                np.log(self.mass_prior(m1, m2))
                            marg_l.append(ml)
                        else:
                            marg_l.append(self.likelihood_floor)
                else:
                    marg_l += len(m2_ax) * [self.likelihood_floor]

        return np.array(marg_l)

    def integrate_likelihood(self, likelihood, axes, measure, log_evidence,
                             mass_array, tov_result):
        """

        """
        l1, l2 = self.eos_model(mass_array, tov_result)

        if l1 is not None:
            marg_likelihood = self.get_marg_likelihood(likelihood, axes,
                                                       l1, l2)
            return logsumexp(marg_likelihood) + np.log(measure) + log_evidence
        else:
            return -np.inf

    def log_likelihood(self, parameters):

        try:
            max_sound_speed = self.parametrization.maximum_speed_of_sound(
                parameters)
        except:
            return -np.inf

        if max_sound_speed <= self.maximum_speed_of_sound:

            try:
                tov_result = self.parametrization.tov_integrate(
                    self.pressure_array, parameters)
            except:
                return -np.inf

            integral = []
            for x in zip(self.likelihood_list, self.axes_list, self.dA_list,
                         self.log_evidence_list, self.mass_array_list):
                integral.append(
                    self.integrate_likelihood(*x, tov_result))
            logL = np.double(sum(np.array(integral)))
            return logL
        else:
            return -np.inf
