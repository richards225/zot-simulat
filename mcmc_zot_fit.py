import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import trapz  # Explicit import

# Mock cosmological data (z, mu, mu_err)
data = np.array([
    [0.1, 35.0, 0.1],
    [0.5, 40.0, 0.2],
    [1.0, 42.5, 0.15],
])
z_data, mu_data, mu_err = data[:, 0], data[:, 1], data[:, 2]

# ZOT dynamic Lambda_eff model
def lambda_eff(tau, Lambda0=1e-5, alpha=1.2e-5, f_L=1.0):
    D_tau = np.exp(-tau)
    pulsar_term = 0.01 * np.sin(2 * np.pi * tau / 10)
    return Lambda0 + alpha * D_tau * f_L + pulsar_term

# Distance modulus for ZOT
def mu_zot(z, H0=70, Om=0.3, Lambda0=1e-5, alpha=1.2e-5):
    def integrand(a, Om, Lambda_eff_a):
        return 1 / np.sqrt(Om / a**3 + Lambda_eff_a + (1 - Om - Lambda_eff_a) / a**2)
    
    mu = []
    c = 3e5  # Speed of light in km/s
    for zi in z:
        a_min = 1 / (1 + zi)
        a_vals = np.linspace(a_min, 1, 1000)
        tau_vals = np.log(a_vals)
        integrands = np.array([integrand(a, Om, lambda_eff(tau, Lambda0, alpha)) for a, tau in zip(a_vals, tau_vals)])
        da = a_vals[1] - a_vals[0]  # Uniform spacing
        chi = trapz(integrands, dx=da)
        DL = (1 + zi) * chi * c / H0  # Mpc
        mu.append(5 * np.log10(DL) + 25)
    return np.array(mu)

# Log-likelihood
def log_likelihood(theta, z, mu, mu_err):
    H0, Om, Lambda0, alpha = theta
    model = mu_zot(z, H0, Om, Lambda0, alpha)
    sigma2 = mu_err**2
    return -0.5 * np.sum((mu - model)**2 / sigma2 + np.log(2 * np.pi * sigma2))  # Full Gaussian

# Log prior
def log_prior(theta):
    H0, Om, Lambda0, alpha = theta
    if 50 < H0 < 100 and 0.1 < Om < 0.5 and 0 < Lambda0 < 1e-4 and 0 < alpha < 1e-4:
        return 0.0
    return -np.inf

# Log posterior
def log_probability(theta, z, mu, mu_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, mu_err)

# Run MCMC
nwalkers = 32
ndim = 4
pos = np.array([70, 0.3, 1e-5, 1.2e-5]) + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(z_data, mu_data, mu_err))
sampler.run_mcmc(pos, 5000, progress=False)

samples = sampler.get_chain(discard=2000, thin=15, flat=True)

# Corner plot
fig = corner.corner(samples, labels=["H0", "Om", "Lambda0", "alpha"], truths=[70, 0.3, 1e-5, 1.2e-5])
plt.savefig('zot_posterior.png')

# Compare chi2
lcdm = FlatLambdaCDM(H0=70, Om0=0.3)
mu_lcdm = 5 * np.log10(lcdm.luminosity_distance(z_data).value) + 25
theta_true = [70, 0.3, 1e-5, 1.2e-5]
chi2_zot = -2 * log_likelihood(theta_true, z_data, mu_data, mu_err)
chi2_lcdm = np.sum((mu_data - mu_lcdm)**2 / mu_err**2)
print(f"ZOT chi^2: {chi2_zot:.2f}, LambdaCDM chi^2: {chi2_lcdm:.2f}")
