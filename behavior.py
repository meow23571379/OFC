def behavior(kx, seed=3):
    d = simulate(kx=kx, seed=seed)
    tru = d["tru"]; est = d["est"]; U = d["U"]
    resid = est[:, x_] - tru[:, x_]
    return dict(true_x_rms=np.sqrt(np.mean(tru[:, x_]**2)),
                resid_rms=np.sqrt(np.mean(resid**2)),
                mean_abs_u=np.mean(np.abs(U[1:])),
                max_abs_u=np.max(np.abs(U[1:])))