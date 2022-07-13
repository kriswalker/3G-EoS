import numpy as np
import bilby
from bilby import result
from likelihood import HyperLikelihood

datafile = 'data/kdes/event_kdes.npz'
npoints = 100
extension = 'eos'

data = np.load(datafile)
densities = data['densities']
logz = []
res = np.shape(densities)[-1]
min_max_values_list = []
events = [0,1,2,3]
inds = events
for i, event in enumerate(events):
    j = inds[i]
    results_file = 'data/json_files/inj_{}_data0_100-0_analysis'.format(event) + \
        '_CECESET1ET2ET3_dynesty_result.json'
    results = result.read_in_result(filename=results_file,
                                    outdir=None, label=None,
                                    extension='json', gzip=False)
    logz.append(results.log_evidence)
    m_ax = data['m_axs'][j]
    l_ax = data['l_axs'][j]
    cm = m_ax[0]
    q = m_ax[1]
    l1 = l_ax[0]
    l2 = l_ax[1]

    mmvl = {}
    mmvl["min_cm"] = min(cm)
    mmvl["max_cm"] = max(cm)
    mmvl["min_q"] = min(q)
    mmvl["max_q"] = max(q)
    mmvl["min_lambda_1"] = min(l1)
    mmvl["max_lambda_1"] = max(l1)
    mmvl["min_lambda_2"] = min(l2)
    mmvl["max_lambda_2"] = max(l2)
    min_max_values_list.append(mmvl)

priors = dict()
priors["log_p"] = bilby.prior.Uniform(32.6, 34.4, name='log_p',
                                      latex_label="$\\log p_0$")
priors["Gamma_1"] = bilby.prior.Uniform(2, 4.5, name='Gamma_1',
                                        latex_label="$\\Gamma_1$")
priors["Gamma_2"] = bilby.prior.Uniform(1.1, 4.5, name='Gamma_2',
                                        latex_label="$\\Gamma_2$")
priors["Gamma_3"] = bilby.prior.Uniform(1.1, 4.5, name='Gamma_3',
                                        latex_label="$\\Gamma_3$")

likelihood = HyperLikelihood(densities, logz, min_max_values_list,
                             number_of_grid_points=res,
                             pressure_array=np.logspace(
                                 np.log10(4e32), np.log10(2.5e35), 100),
                             maximum_mass=1.97,
                             maximum_speed_of_sound=1.1,
                             log_p=priors['log_p'],
                             Gamma_1=priors['Gamma_1'],
                             Gamma_2=priors['Gamma_2'],
                             Gamma_3=priors['Gamma_3']
                             )

results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    npoints=npoints, verbose=False, resume=False,
    outdir=homedir + 'results/outdir_n{0}_{1}'.format(npoints, extension))
results.plot_corner()
