# ----------------------------------------------------------------- 真值產生用的抽樣
# 為了在「產生真值」時能對 T 加世界擾動、對 x 加感官雜訊 ω，
# 我們直接抽兩個獨立高斯量: w_T ~ N(0,qT), ω ~ N(0,qx)。
def simulate(kx=g, seed=3, return_traces=True):
    rng = np.random.default_rng(seed)
    A = make_A(kx)
    control_u = lambda xhat: A @ xhat

    xt   = np.zeros((N, n)); xt[0, T_] = 5.0; xt[0, x_] = 5.0
    xhat = np.zeros(n);      xhat[T_] = 5.0;  xhat[x_] = 5.0
    P    = np.diag([sigma_T2, 1., 1., 1., 1., 1.])

    est = np.zeros((N, n)); tru = np.zeros((N, n))
    Kh  = np.zeros((N, n)); Ptr = np.zeros(N); Pxx = np.zeros(N)
    Utrace = np.zeros(N)

    for k in range(N):
        # -------- 產生真實 state（含控制 u 與雜訊）-----------------
        if k > 0:
            u = control_u(xhat)                       # u = A·x̂，6 維向量
            # 被動演化 + 控制輸入 (g 已從 Φ 移出，改由 u = A·x̂ 進入 r'')
            xt[k] = Phi @ xt[k-1] + u
            # 世界擾動: T 加 random-walk 噪音
            wT = np.sqrt(qT) * rng.standard_normal()
            xt[k, T_] = xt[k-1, T_] + wT              # T 是純 random walk
            # v11: 真值 x 保持乾淨的物理量 (T-H)，不再加 ω。
            # 感官雜訊 ω 改在量測那一刻注入 (見下方 z)。
            xt[k, x_] = xt[k, T_] - xt[k, H_]
            Utrace[k] = u[rpp_]

        # -------- Kalman 量測更新 (量 x) ---------------------------
        # v11: 感官雜訊 ω 在這裡注入 (用真實 σ_ω²)，但 filter 內部用 R→0。
        omega = np.sqrt(sigma_omega2) * rng.standard_normal(1)
        z = H_meas @ xt[k] + omega
        S = H_meas @ P @ H_meas.T + R
        K = (P @ H_meas.T @ np.linalg.inv(S)).ravel()
        xhat = xhat + K * (z - H_meas @ xhat)
        P = (I - np.outer(K, H_meas.ravel())) @ P

        est[k] = xhat; tru[k] = xt[k]
        Kh[k] = K; Ptr[k] = np.trace(P); Pxx[k] = P[x_, x_]

        # -------- Kalman 時間更新 (預測) ---------------------------
        xhat = Phi @ xhat + control_u(xhat)
        P = Phi @ P @ Phi.T + Q

    if return_traces:
        return dict(t=np.arange(N)*dt, est=est, tru=tru, Kh=Kh,
                    Ptr=Ptr, Pxx=Pxx, U=Utrace)
    return None