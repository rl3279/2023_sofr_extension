import rfr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def rfr_bond_rec(
    vol: np.ndarray=None,
    Ts: np.ndarray=None,
    rhos: np.ndarray=None,
    T: float=None,
    n: int = 1000,
    r0: np.ndarray = 0,
    p0: np.ndarray=1,
    dt: float = 1e-4,
    R: np.ndarray = None
):
    if R is None:
        try:
            _, R = rfr.sim_r_paths_Q(vol, Ts, rhos, T, n, r0, dt)
        except:
            raise RuntimeError(
                ("No RFR tensor passed, nor are specified arguments "
                 "valid to construct RFR samples."
                )
            )
    taus = np.diff(Ts)
    J, n, M = R.shape
    P = np.zeros((J+1,n,M))
    P[0,:,:] = p0
    for j in range(1, J+1):
        P[j,:,:] = P[j-1,:,:] / (1 + taus[j-1] * R[j-1,:,:])
    return P

def rfr_futures_mc(
    vol: np.ndarray,
    Ts: np.ndarray,
    rhos: np.ndarray,
    T: float,
    n: int = 1000,
    r0: float = 0,
    dt: float = 1e-4,
    R: np.ndarray = None
):
    if R is None:
        try:
            _, R = rfr.sim_r_paths_Q(vol, Ts, rhos, T, n, r0, dt)
        except:
            raise RuntimeError(
                ("No RFR tensor passed, nor are specified arguments "
                 "valid to construct RFR samples."
                )
            )
    J, n, M = R.shape
    M -= 1
    futures = R.mean(axis = 1)
    convexity = (R - futures.reshape(J, 1, M+1)).mean(axis=1)
    return convexity, futures

def rfr_vanilla_swap_mc(
    vol: np.ndarray,
    Ts: np.ndarray,
    rhos: np.ndarray,
    T: float,
    n: int = 1000,
    r0: np.ndarray = 0,
    p0: float=1,
    dt: float = 1e-4,
    R: np.ndarray = None
):
    if R is None:
        try:
            _, R = rfr.sim_r_paths_Q(vol, Ts, rhos, T, n, r0, dt)
        except:
            raise RuntimeError(
                ("No RFR tensor passed, nor are specified arguments "
                 "valid to construct RFR samples."
                )
            )

    J, n, M = R.shape
    Ts = Ts.reshape(-1, 1)
    M -= 1 
    if vol.shape[1] == M:
        vol = np.append(vol, np.zeros((vol.shape[0], 1)), axis = 1)
    elif vol.shape[1] == M+1:
        pass
    else:
        raise RuntimeError("vol invalid shape.")

    disc_bond = rfr_bond_rec(
        vol=vol[:,:-1], Ts=Ts.flatten(),
        rhos=rhos, T=T, n=n, r0=0,
        p0=p0, dt=dt
    )
    Swaplet = np.zeros((J, n, M))
    idx = (Ts / dt).astype(int).flatten()
    for j in range(J):
        fra = disc_bond[j+1,:,:idx[j+1]] * (R[j,:,:idx[j+1]] - K)
        np.copyto(Swaplet[j,:,:idx[j+1]], fra)
        # fill with last realized value
        Swaplet[j,:,idx[j+1]:] = Swaplet[j,:,idx[j+1]-1].reshape(-1,1)

    Swap = Swaplet.sum(axis=0)
    Price = Swap.mean(axis=0)
    return Swap, Swaplet, Price

def Black(
    Rj:np.ndarray,
    K:np.ndarray,
    v:np.ndarray
):
    # R (J, n, M+1)
    # v (J, M+1)
    Phi = stats.norm.cdf
    d1 = (np.log(Rj / K) + v / 2) / (np.sqrt(v))
    # print("d1", d1)
    # print(Phi(d1))
    d2 = d1 + np.sqrt(v)
    # print("d2", d2)
    # print(Phi(d2))
    return Rj * Phi(d1) - K * Phi(d2)

def rfr_cap_mc(
    vol: np.ndarray,
    Ts: np.ndarray,
    rhos: np.ndarray,
    T: float,
    # kind: str= "f",
    K: float = 1,
    n: int = 1000,
    p0: float=1,
    r0: np.ndarray = np.log(1e-2),
    dt: float = 1e-3,
    R: np.ndarray = None
):
    if R is None:
        try:
            _ , R = zip(*[rfr.sim_r_paths_T1(vol[j], Ts[j], Ts[j+1], T, n, r0, dt, lognormal=True)
                 for j in range(len(Ts)-1)])
            # dR = np.array(dR)
            R = np.array(R)
        except:
            raise RuntimeError(
                ("No RFR tensor passed, nor are specified arguments "
                 "valid to construct RFR samples."
                )
            )

    taus = np.diff(Ts)
    J, n, M = R.shape
    Ts = Ts.reshape(-1, 1)
    M -= 1 
    if vol.shape[1] == M:
        vol = np.append(vol, np.zeros((vol.shape[0], 1)), axis = 1)
    elif vol.shape[1] == M+1:
        pass
    else:
        raise RuntimeError("vol invalid shape.")

    t = np.arange(0, T+dt, dt)
    vf = vol**2 * ((Ts[:-1]-t)) # -t? 
    vb = vol**2 * (
        np.maximum(
            Ts[:-1]-t, 0
        ) + (
            (Ts[1:] - np.maximum(Ts[:-1], t))**3 / (taus**2 * 3).reshape(-1, 1)
        )
    )
    # vf, vb shapes (3, 1001) = (J, M+1)
    # disc_bond = rfr_bond_rec(Ts=Ts.flatten(), R=R)
    disc_bond = rfr_bond_rec(
        vol=vol[:,:-1], Ts=Ts.flatten(),
        rhos=rhos, T=T, n=n, r0=0,
        p0=p0, dt=dt
    )
    # rfr.debug("disc_bond", disc_bond)
    idx = (Ts / dt).astype(int).flatten()

    Caplet = np.zeros((2, J, n, M))
    for j in range(J):

        Cf = disc_bond[j+1,:,:idx[j]] * Black(R[j,:,:idx[j]], K, vf[j][:idx[j]]) # (n, Tj-1 / dt + 1)
        Cb = disc_bond[j+1,:,:idx[j+1]] * Black(R[j,:,:idx[j+1]], K, vb[j][:idx[j+1]]) # (n, Tj / dt + 1)
        np.copyto(Caplet[0,j,:,:idx[j]], Cf)
        np.copyto(Caplet[1,j,:,:idx[j+1]], Cb)
        # fill with last realized value
        Caplet[0,j,:,idx[j]:] = Caplet[0,j,:,idx[j]-1].reshape(-1,1)
        Caplet[1,j,:,idx[j+1]:] = Caplet[1,j,:,idx[j+1]-1].reshape(-1,1)
    
    Cap = Caplet.sum(axis=1)
    Price = Cap.mean(axis=1)
    return Cap, Caplet, Price





if __name__ == "__main__":

    m = 1000
    n = 1000
    vol = np.array([
        [0.02] * m, 
        [0.04] * m, 
        [0.08] * m
    ])

    Ts = np.array([0., 0.3, 0.4, 0.8])
    rhos = rfr.random_corr_mat(full_random=True, n=3, seed=41)
    r0 = 0.01
    T = 1
    dt = T/m
    K = 0.01
    # Cap, Caplet, Price = rfr_cap_mc(vol, Ts, rhos, T, K, n, r0, dt)
    Swap, Swaplet, Price = rfr_vanilla_swap_mc(vol, Ts , rhos, T, n, r0, 1, dt)
    rfr.debug("Swap", Swap)
    rfr.debug("Swaplet", Swaplet)
    rfr.debug("Price", Price)
    
    fig, ax = plt.subplots(1,2,figsize=(10,7))
    ax[0].plot(Swap.T)
    ax[1].plot(Price)
    plt.show()
    # for i in range(2):
    #     for j in range(2):
    #         target = Price[i] if j else Swap[i].T
    #         ax[i][j].plot(np.arange(0, T, 1/m), target)
    #         ax[i][j].set_xlabel("T")
    #         ylabel = [x + "backward" if i else x + "forward" for x in ["Cap_", "Cap_Price_"]]
    #         ax[i][j].set_ylabel(ylabel[j])

    # ax[1][1].plot(np.arange(0, T, 1/m), Price[0], color = "red", linestyle="--")
    # for i in range(2):
    #     for j in range(2):
    #         for t in range(len(Ts)):
    #             ax[j][i].axvline(x=Ts[t], color="black", linestyle = "--")
    # plt.show()
