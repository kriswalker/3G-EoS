import sys
import os
import dill
import numpy as np
from bilby import result
import dynesty
from dynesty import plotting as dyplot
from schwimmbad import MPIPool
import matplotlib.pyplot as plt

import spectral_decomposition
from likelihood import EOSLikelihood

dynesty.utils.pickle_module = dill

multiprocess = int(sys.argv[1])
resume = int(sys.argv[2])
N = int(sys.argv[3])

if multiprocess:
    pool = MPIPool(use_dill=True)
    npool = pool.size + 1
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
else:
    pool = None
    npool = None

# =============================================================================
#  data options
# =============================================================================

json_dir = '/data/json_files/many_events'
kdefile = '/data/kdes/event_kdes_manyevents_25px.npz'

# =============================================================================
#  physics options
# =============================================================================

eos = 'spectral'
pressure_array = np.logspace(np.log10(4e32), np.log10(2.5e35), 100)
maximum_mass = 1.97
maximum_speed_of_sound = 1.1

# =============================================================================
#  sampler options
# =============================================================================

nlive = 128
verbose = True

label = 'eos'
outdir = '/results/outdir_manyevents_{}loudest_spectral'.format(N)

# =============================================================================
# =============================================================================
# =============================================================================

data = np.load(kdefile, allow_pickle=True)
# randinds = np.random.choice(np.arange(76), N, replace=False)
events = data['event_ids'][:N]
kde_list = data['kdes'][:N]
axes_list = data['axes'][:N]
del data

print('Performing equation of state parameter estimation using events', events)
print('KDEs loaded from ' + kdefile)
print('Parametrization: ' + eos)
print('Min, max central pressure:', min(pressure_array), max(pressure_array))
print('Maximum mass:', maximum_mass)
print('Maximum speed of sound:', maximum_speed_of_sound)

print('Reading in log evidence for each injection...')
logZ_list = []
for i, event in enumerate(events):
    results_file = json_dir + '/inj_{}_data0_100-0_analysis_CECESET1ET2ET3_' \
        'dynesty_result.json'.format(event)
    results = result.read_in_result(filename=results_file,
                                    extension='json', gzip=False)
    logZ_list.append(results.log_evidence)
print('log evidence:', logZ_list)


def prior_transform(u):

    # gamma_0, gamma_1, gamma_2, gamma_3
    lows = np.array([0.2, -1.6, -0.6, -0.02])
    highs = np.array([2.0, 1.7, 0.6, 0.02])

    return lows + u * (highs - lows)


ndims = 4
parametrization = spectral_decomposition

param_labels = [r'$\gamma_0$', r'$\gamma_1$', r'$\gamma_2$', r'$\gamma_3$']


def mass_prior(m1, m2):
    return 1.0


likelihood = EOSLikelihood(
    kde_list, axes_list, logZ_list, pressure_array, maximum_mass,
    maximum_speed_of_sound, mass_prior, parametrization,
    likelihood_floor=-1e3)

if resume:
    print('Resuming from', outdir + '/{}_resume.pickle'.format(label))
    sampler = dynesty.NestedSampler.restore(
        outdir + '/{}_resume.pickle'.format(label), pool=pool)
    sampler.run_nested(resume=True)
else:
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    print('Beginning hyperparameter estimation. Outputs will be stored in '
          '{}'.format(outdir))
    sampler = dynesty.NestedSampler(
        likelihood.log_likelihood, prior_transform, ndims, nlive=nlive,
        pool=pool)
    sampler.run_nested(
        checkpoint_file=outdir+'/{}_resume.pickle'.format(label),
        checkpoint_every=600, print_progress=verbose)

print('Hyperparameter estimation complete. Saving to file...')
dynesty.utils.save_sampler(
    sampler, outdir + '/{}_dynesty.pickle'.format(label))

fig_, axes_ = plt.subplots(ndims, ndims, figsize=(10, 10))
fig, axes = dyplot.cornerplot(
    sampler.results, color='dodgerblue', labels=param_labels,
    show_titles=True, fig=(fig_, axes_))
fig.savefig(outdir + '/{}_corner.png'.format(label), dpi=300)
plt.close(fig)

config_file = open(outdir + '/config_options.txt', 'w')
labels = ['eos', 'events', 'likelihood file', 'pressure array', 'maximum mass',
          'maximum sound speed', 'sampler', 'nlive', 'npool']
vals = [eos, events, kdefile, pressure_array, maximum_mass,
        maximum_speed_of_sound, sampler, nlive, npool]
for label, val in zip(labels, vals):
    config_file.write('{} {}\n'.format(label, val))
config_file.close()
