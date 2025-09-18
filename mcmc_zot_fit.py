import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import minimize

# Mock cosmological data (e.g., redshift z and distance modulus mu; replace with real Planck/DESI CSV)
# Format: z, mu, mu_err
data = np.array([
    [0.1, 35.0, 0.1],
    [0.5, 40.0, 0.2],
    [1.0, 42.5, 0.15],
    # Add more rows or load from CSV: data = np.loadtxt('data/planck_mock.csv', delimiter=',')
])

z_data, mu_data, mu_err = data[:, 0], data[:, 1], data[:, 2]

# ZOT dynamic Lambda_eff model
def lambda_eff(tau, Lambda0=1e-5, alpha=1.2e-5, f_L=1.0):
    """Dynamic Lambda_eff with Locksmith modulation and Higgs-Pulsar oscillation.
    tau: conformal time proxy (e.g., log(a), where a is scale factor).
    In full ZOT, incorporate Postulate 3: oscillatory term ~ sin(omega * tau).
    """
    D_tau = np.exp(-tau)  # Simplified deviation term
    pulsar_term = 0.01 * np.sin(2 * np.pi * tau / 10)  # Higgs-Pulsar oscillation (tune omega)
    return Lambda0 + alpha * D_tau * f_L + pulsar_term

# Distance modulus for ZOT cosmology (simplified Friedmann integration)
def mu_zot(z, H0=70, Om=0.3, Lambda0=1e-5, alpha=1.2e-5):
    """Compute mu(z) under ZOT. Integrate 1/H(a) for comoving distance."""
    def integrand(a, Om, Lambda_eff_a):
        return 1 / np.sqrt(Om / a**3 + Lambda_eff_a + (1 - Om - Lambda_eff_a) / a**2)
    
    mu = []
    for zi in z:
        tau_i = np.log(1 / (1 + zi))  # Simplified tau ~ -ln(a)
        Lambda_eff_i = lambda_eff(tau_i, Lambda0, alpha)
        integrals = [integrand(a, Om, lambda_eff(np.log(a), Lambda0, alpha)) for a in np.linspace(1/(1+zi), 1, 1000)]
        chi = np.trapz(integrals, dx=1/len(integrals))
        mu.append(5 * np.log10((1 + zi) * chi * 3e5 / H0) + 25)  # Approx DL in Mpc
    return np.array(mu)

# Log-likelihood for MCMC
def log_likelihood(theta, z, mu, mu_err):
    H0, Om, Lambda0, alpha = theta
    model = mu_zot(z, H0, Om, Lambda0, alpha)
    sigma2 = mu_err**2
    return -0.5 * np.sum((mu - model)**2 / sigma2 + np.log(sigma2))

# Prior (uniform for simplicity)
def log_prior(theta):
    H0, Om, Lambda0, alpha = theta
    if 50 < H0 < 100 and 0.1 < Om < 0.5 and 0 < Lambda0 < 1e-4 and 0 < alpha < 1e-4:
        return 0.0
    return -np.inf

# Posterior
def log_probability(theta, z, mu, mu_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu, mu_err)

# Run MCMC
nwalkers = 32
ndim = 4  # Parameters: H0, Om, Lambda0, alpha
pos = [70, 0.3, 1e-5, 1.2e-5] + 1e-4 * np.random.randn(nwalkers, ndim)  # Initial guess

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(z_data, mu_data, mu_err))
sampler.run_mcmc(pos, 5000, progress=True)  # Burn-in + samples

# Flatten samples (discard first 2000 as burn-in)
samples = sampler.get_chain(discard=2000, thin=15, flat=True)

# Plot corner plot
fig = corner.corner(samples, labels=["H0", "Om", "Lambda0", "alpha"], truths=[70, 0.3, 1e-5, 1.2e-5])
plt.savefig('results/zot_posterior.png')

# Compare to LambdaCDM
lcdm = FlatLambdaCDM(H0=70, Om0=0.3)
mu_lcdm = 5 * np.log10(lcdm.luminosity_distance(z_data).value) + 25
chi2_zot = -2 * log_likelihood([70, 0.3, 1e-5, 1.2e-5], z_data, mu_data, mu_err)
chi2_lcdm = np.sum((mu_data - mu_lcdm)**2 / mu_err**2)
print(f"ZOT chi^2: {chi2_zot:.2f}, LambdaCDM chi^2: {chi2_lcdm:.2f}")

# TODO: Expand with real data, Locksmith function details, non-flat geometry, or JWST integrations.