# zot_extensions.py
# Extensions for ZOT N-Body repositor 
# - dynamic ZOT Matrix (S) with controlled update schedule  - ZOT Zero Oporator Theory  - Ricardo Bartolome -SP Brazil.
# - Benettin-style Lyapunov exponent estimator (robust)
# - Monitoring of conserved quantities (momentum, angular momentum, energy components)
# - Runner that integrates in chunks and updates S adaptively
#
# Usage:
# - Import functions into your repo: from zot_extensions import *
# - Use `run_with_dynamic_S(...)` instead of a single solve_ivp call
# - Use `estimate_lyapunov_benettin(...)` for robust lambda_max

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw
import time

G = 1.0

# ---------------- Locksmith / D(tau) ----------------
def locksmith_function(tau, k=0.0, c=1.0, delta=1e-6):
    if tau <= 0:
        return 0.0
    arg = tau * np.exp(np.clip(k * tau, -700, 700))
    W = lambertw(arg).real if arg > 0 else 0.0
    sigmoid = 1.0 / (1.0 + np.exp(-c * (tau - delta)))
    return float(tau * W * sigmoid)

def D_of_tau_dynamic(tau, alpha=1.0, k=0.0, c=1.0, delta=1e-6):
    return alpha * locksmith_function(tau, k, c, delta)

# ---------------- pairwise and compression utilities ----------------
def pair_force(r_i, r_j, eps):
    dr = r_i - r_j
    r2 = np.dot(dr, dr) + eps * eps
    r = np.sqrt(r2)
    return -dr / (r2 * r)


def Vc_and_grad(positions, mu=1e-3, eps=1e-4):
    n = positions.shape[0]
    V = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[i] - positions[j]
            r2 = np.dot(dr, dr) + eps * eps
            r = np.sqrt(r2)
            V += mu / r
            g = -mu * dr / (r2 * r)
            grad[i] += g
            grad[j] -= g
    return float(V), grad

# ---------------- ZOT Matrix generation (improved) ----------------
def generate_zot_matrix(positions, threshold=1.0, sparsity=0.5, keep_self=True):
    """Generate symmetric ZOT sparsity matrix S based on distances.
    - positions: (n,3)
    - threshold: keep all pairs with r < threshold
    - sparsity: probability to KEEP a given pair (so higher sparsity -> more kept)
    Returns S (n,n) with ones for kept pairs and zeros else; diagonal is zero.
    """
    n = positions.shape[0]
    S = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            # deterministic keep if within threshold; else stochastic by sparsity
            if rij < threshold or np.random.rand() < sparsity:
                S[i, j] = 1
                S[j, i] = 1
    if keep_self:
        np.fill_diagonal(S, 0)
    return S

# ---------------- RHS that uses S (unchanged physics) ----------------
def zot_rhs_givenS(tau, y, masses, params, S):
    n = len(masses)
    pos = y[:3*n].reshape((n, 3))
    vel = y[3*n:].reshape((n, 3))
    acc = np.zeros_like(pos)

    D = D_of_tau_dynamic(tau, params.get('alpha', 1.0), params.get('k', 0.0), params.get('c', 1.0), params.get('delta', 1e-6))
    eps = params.get('eps0', 1e-4) * (1.0 + params.get('beta', 1e-2) * D)
    kappa = params.get('kappa', 1e-3)
    mu = params.get('mu', 1e-3)
    xi0 = params.get('xi0', 1e-4)
    xi = xi0 * D

    # gravitational part masked by S
    for i in range(n):
        for j in range(n):
            if i == j or S[i, j] == 0:
                continue
            acc[i] += G * masses[j] * pair_force(pos[i], pos[j], eps)

    Vc, gradVc = Vc_and_grad(pos, mu=mu, eps=eps)
    acc += - (kappa * D) * gradVc
    acc += - (xi / masses[:, None]) * vel

    dydt = np.zeros_like(y)
    dydt[:3 * n] = vel.flatten()
    dydt[3 * n:] = acc.flatten()
    return dydt

# ---------------- Adaptive S-runner ----------------
def run_with_dynamic_S(masses, pos0, vel0, params, t_span=(0.0,10.0),
                       dt_update=0.5, update_on_close_factor=1.5, verbose=True):
    """Integrate the system in chunks and update ZOT matrix S adaptively.
    - dt_update: time chunk length between S recomputations
    - update_on_close_factor: if min_pair_dist < update_on_close_factor * eps then force immediate recompute
    Returns a dict with solution arrays concatenated and diagnostics.
    """
    n = len(masses)
    y0 = np.concatenate([pos0.flatten(), vel0.flatten()])
    t0, tf = t_span
    # prepare outputs
    t_all = []
    y_all = []
    current_t = t0
    y_current = y0.copy()

    # initial S
    eps0 = params.get('eps0', 1e-4)
    S = generate_zot_matrix(pos0, threshold=params.get('threshold', 1.0), sparsity=params.get('sparsity', 0.5))

    start_time = time.time()
    while current_t < tf:
        t_end = min(current_t + dt_update, tf)
        sol = solve_ivp(lambda t,y: zot_rhs_givenS(t,y,masses,params,S), (current_t, t_end), y_current,
                        rtol=params.get('rtol',1e-9), atol=params.get('atol',1e-12), max_step=params.get('max_step',0.2))
        # append
        t_all.append(sol.t)
        y_all.append(sol.y)
        # update current
        current_t = sol.t[-1]
        y_current = sol.y[:, -1]
        # decide whether to recompute S: check min distance
        pos_now = y_current[:3*n].reshape((n,3))
        min_pair = np.min([np.linalg.norm(pos_now[i]-pos_now[j]) for i in range(n) for j in range(i+1,n)]) if n>1 else np.inf
        eps_now = eps0 * (1.0 + params.get('beta',1e-2) * D_of_tau_dynamic(current_t, params.get('alpha',1.0), params.get('k',0.0), params.get('c',1.0), params.get('delta',1e-6)))
        # recompute if pairs getting close or periodic update
        if min_pair < update_on_close_factor * eps_now or (current_t - t0) % (5*dt_update) < 1e-12:
            S = generate_zot_matrix(pos_now, threshold=params.get('threshold',1.0), sparsity=params.get('sparsity',0.5))
        # continue loop
    elapsed = time.time() - start_time
    # concatenate arrays
    t_full = np.concatenate(t_all)
    y_full = np.concatenate(y_all, axis=1)
    # diagnostics
    pos_final = y_full[:3*n, -1].reshape((n,3))
    vel_final = y_full[3*n:, -1].reshape((n,3))
    energy_start = compute_energy(pos0, vel0, masses, params, t0)
    energy_end = compute_energy(pos_final, vel_final, masses, params, t_full[-1])
    # momentum & angular momentum
    P0 = np.sum(masses[:,None]*pos0, axis=0)
    L0 = np.sum(np.cross(pos0, masses[:,None]*vel0), axis=0)
    P1 = np.sum(masses[:,None]*pos_final, axis=0)
    L1 = np.sum(np.cross(pos_final, masses[:,None]*vel_final), axis=0)

    result = {
        't': t_full,
        'y': y_full,
        'elapsed': elapsed,
        'energy_start': energy_start,
        'energy_end': energy_end,
        'momentum_start': P0.tolist(),
        'momentum_end': P1.tolist(),
        'angmom_start': L0.tolist(),
        'angmom_end': L1.tolist()
    }
    if verbose:
        print(f"run_with_dynamic_S finished: points={y_full.shape[1]}, elapsed={elapsed:.2f}s")
        print("Eeff start", energy_start['Eeff'], "end", energy_end['Eeff'])
        print("L start", L0, "L end", L1)
    return result

# ---------------- Benettin Lyapunov estimator ----------------

def estimate_lyapunov_benettin(masses, pos0, vel0, params, S_func=None, t_total=10.0, dt_renorm=0.1, delta0=1e-8):
    """Estimate maximal Lyapunov exponent via Benettin method.
    - S_func: function(positions, params) -> S matrix; if None we use static generate_zot_matrix from pos0
    - dt_renorm: time interval between renormalizations
    Returns lambda_est and diagnostic time series.
    """
    n = len(masses)
    y_ref = np.concatenate([pos0.flatten(), vel0.flatten()])
    # perturbed initial condition (small relative position perturbation)
    y_pert = y_ref.copy()
    y_pert[0] += delta0

    t = 0.0
    renorms = 0
    sum_log = 0.0
    times = []
    lambda_vals = []

    # prepare initial S
    if S_func is None:
        S = generate_zot_matrix(pos0, threshold=params.get('threshold',1.0), sparsity=params.get('sparsity',0.5))
        def S_func_dummy(pos,params):
            return S
        S_func = S_func_dummy

    while t < t_total:
        t_end = min(t + dt_renorm, t_total)
        # integrate both trajectories over dt_renorm
        sol_ref = solve_ivp(lambda tt,yy: zot_rhs_givenS(tt, yy, masses, params, S_func(y_ref[:3*n].reshape((n,3)), params)), (t, t_end), y_ref,
                            t_eval=[t_end], rtol=1e-9, atol=1e-12)
        sol_pert = solve_ivp(lambda tt,yy: zot_rhs_givenS(tt, yy, masses, params, S_func(y_pert[:3*n].reshape((n,3)), params)), (t, t_end), y_pert,
                             t_eval=[t_end], rtol=1e-9, atol=1e-12)
        y_ref = sol_ref.y[:, -1]
        y_pert = sol_pert.y[:, -1]
        # compute separation
        sep_vec = y_pert - y_ref
        sep = np.linalg.norm(sep_vec)
        if sep == 0:
            sep = 1e-20
        # renormalize perturbed trajectory to delta0 distance along sep_vec direction
        renorms += 1
        sum_log += np.log(sep / delta0)
        # set new perturbed state
        sep_dir = sep_vec / sep
        y_pert = y_ref + sep_dir * delta0
        t = t_end
        times.append(t)
        lambda_vals.append(sum_log / (t if t>0 else 1e-20))
    lambda_est = sum_log / (t_total if t_total>0 else 1e-20)
    return {'lambda_est': float(lambda_est), 'times': np.array(times), 'lambda_vals': np.array(lambda_vals)}

# ---------------- energy comp reused ----------------
def compute_energy(pos, vel, masses, params, tau):
    n = len(masses)
    D = D_of_tau_dynamic(tau, params.get('alpha', 1.0), params.get('k', 0.0), params.get('c', 1.0), params.get('delta', 1e-6))
    eps = params.get('eps0', 1e-4) * (1.0 + params.get('beta', 1e-2) * D)
    K = 0.5 * np.sum(masses[:, None] * vel ** 2)
    U = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(pos[i] - pos[j])
            U += - G * masses[i] * masses[j] / np.sqrt(rij**2 + eps**2)
    Vc, _ = Vc_and_grad(pos, mu=params.get('mu', 1e-3), eps=eps)
    Eeff = K + U + params.get('kappa', 1e-3) * D * Vc
    return {'K': float(K), 'U': float(U), 'Vc': float(Vc), 'Eeff': float(Eeff)}

# ---------------- Example runner ----------------
def example_run():
    # simple example showing how to use run_with_dynamic_S and estimate_lyapunov_benettin
    n = 3
    masses = np.ones(n)
    pos0 = np.array([[0.0,0.0,0.0],[1.0,0.0,0.0],[0.3,0.8,0.0]])
    vel0 = np.array([[0.0,0.0,0.0],[0.0,0.5,0.0],[-0.4,-0.1,0.0]])
    params = {'eps0':1e-4,'beta':1e-2,'kappa':1e-3,'mu':1e-3,'xi0':1e-4,'alpha':1.0,'k':0.0,'c':1.0,'delta':1e-6,'threshold':2.0,'sparsity':0.3,'rtol':1e-9,'atol':1e-12,'max_step':0.2}
    res = run_with_dynamic_S(masses, pos0, vel0, params, t_span=(0.0,10.0), dt_update=0.5, update_on_close_factor=1.5)
    ly = estimate_lyapunov_benettin(masses, pos0, vel0, params, t_total=10.0, dt_renorm=0.1, delta0=1e-8)
    print('Example run done. E0, E1:', res['energy_start']['Eeff'], res['energy_end']['Eeff'])
    print('Lyapunov estimate (Benettin):', ly['lambda_est'])

if __name__ == '__main__':
    example_run()
