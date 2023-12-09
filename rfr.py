import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def rfr_decay(T0, T1, t):
    return np.minimum(np.maximum(T1 - t, 0) / (T1 - T0), 1)


def sim_r_paths_T1(vol, T0, T1, T, n=1000, r0=0, dt=1e-4, lognormal:bool=False):
    m = int(T / dt)
    if not isinstance(vol, float):
        if len(vol) != m:
            raise ValueError(
                f"If input dynamic volatility, then vol must equal simulation length. len(vol)={len(vol)}, m = {m}."
            )
    else:
        vol = np.ones(m) * vol
    t = np.arange(0, T, dt)
    G = rfr_decay(T0, T1, t)
    Z = np.random.randn(n, m)
    dR = np.multiply(Z, vol * G * np.sqrt(dt))
    R = dR.cumsum(axis = 1) + r0
    R = np.hstack((np.ones((n,1))*r0, R))
    if lognormal: 
        R = np.exp(R)
    return dR, R


def sim_r_paths_Q(
    vol: np.ndarray,
    Ts: np.ndarray,
    rhos: np.ndarray,
    T: float,
    n: int = 1000,
    r0: float = 0,
    dt: float = 1e-4,
):
    """
    RFR rate under Q risk neutral measure for P accrual periods.


    Parameters:
    --------------
    vol: np.ndarray: volatility matrix of shape (J, m). Each colums is
    the discrete volatility function for accrual period i.

    Ts: Iterable: list of accrual times. i.e. [T_0, T_1, ..., T_J]

    rhos: np.ndarray: correlation matrix between Wiener processes associated
    with different accrual periods.

    T: float: length of simulation period. Must be greater than Ts[-1].

    n: int: number of samples

    r0: np.ndarray: vector of starting rates for each accrual period

    dt: float: time increment


    Returns:
    --------------
    np.ndarray: a tensor of dimension (#periods, #samples, #time points) containing
    RFR increments dRj for each period (Tj-1, Tj).
    """
    # make correlation lower triangular for better vectorization
    rhos = np.tril(rhos)
    M = int(T / dt)
    # calculate accrual times
    taus = np.diff(Ts)
    J = vol.shape[0]
    dR = np.zeros((J, n, M))
    R = np.zeros((J, n, M+1))
    for j in range(1, J + 1):  # for each accrual period
        # initialize dRj, Rj and drift
        dRj = np.zeros((n, M))
        Rj = np.zeros((n, M + 1))
        Rj[:, 0] = r0
        alpha = np.zeros((n, M))
        # compute diffusion term (martingale)
        dWj, _ = sim_r_paths_T1(vol[j - 1, :], Ts[j - 1], Ts[j], T, n, r0, dt)
        # compute decay function used in drift
        gj = rfr_decay(Ts[j - 1], Ts[j], np.arange(0, T, dt))

        for m in range(M):  # for each time increment
            # vectorized decay function
            decay = np.array(
                [rfr_decay(Ts[i - 1], Ts[i], m * dt) for i in range(1, J + 1)]
            )

            # vectorized previous accrual effects
            prev_period_term = (taus * vol[:, m] * decay).reshape(-1, 1) / (
                1 + taus.reshape(-1, 1) * Rj[:, m]
            )
            # since correlation matrix is lower triangular, can use inner product
            # thus all periods after j has 0 effect
            np.copyto(
                alpha[:, m], vol[j - 1, m] * gj[m] * (rhos[j - 1] @ prev_period_term) * dt
            )
            # Rj increment given by drift + diffusion
            np.copyto(dRj[:, m], alpha[:, m] + dWj[:, m])
            # Rj level given by previous level + increment
            np.copyto(Rj[:, m+1], Rj[:, m] + dRj[:, m])

        np.copyto(dR[j - 1], dRj)
        np.copyto(R[j-1], Rj)
    return dR, R


def debug(name="Var", *arg):
    for a in arg:
        if hasattr(a, "shape"):
            printout = a.shape
        else:
            printout = a
        if hasattr(a, "__name__"):
            name = a.__name__
        else:
            name = name
        print()
        print(f"{name}: {printout}, type: {type(a)}")
        print()


def random_corr_mat(
    full_random=False, n: int = 2, eigs: np.ndarray = None, seed=None, decay=False
):
    if not full_random:
        assert eigs is not None
    else:
        np.random.seed(seed)
        eigs = np.random.random(n)
        if decay:
            eigs = np.cumprod(eigs)
        eigs = eigs / eigs.sum() * n
    return stats.random_correlation.rvs(eigs, seed)


if __name__ == "__main__":

    m = 1000
    n = 10
    vol = np.array([
        [0.02] * m, 
        [0.04] * m, 
        [0.0] * m
    ])
    Ts = np.array([0.2, 0.3, 0.4, 0.8])
    rhos = random_corr_mat(full_random=True, n=3)
    r0 = np.array([0, 0, 0])
    T = 1
    
    dR, R = sim_r_paths_Q(vol, Ts, rhos, T, n=n, r0=r0, dt=1/m)
    # J, n, m = dR.shape
    # print(f"dR shape: J = {J}, n = {n}, m = {m}")
    J, n, m = R.shape
    
    fig, ax = plt.subplots(J, 1, figsize=[10, J * 5 * 1.1], sharey=True)
    for j in range(J):
        ax[j].plot(np.arange(0, T, 1/m), R[j].T)
        ax[j].axvline(x=Ts[j], color="black", linestyle = "--")
        ax[j].axvline(x=Ts[j+1], color="black", linestyle = "--")

    plt.show()
