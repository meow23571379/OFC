import numpy as np
from make_A import *
from config import *
# ----------------------------------------------------------------- 真值產生用的抽樣
# 為了在「產生真值」時能對 T 加世界擾動、對 x 加感官雜訊 ω，
# 我們直接抽兩個獨立高斯量: w_T ~ N(0,qT), ω ~ N(0,qx)。
def simulate(kx=6, seed=3, return_traces=True):
    rng = np.random.default_rng(seed)
    A = make_A(kx, n)
    control_u = lambda xhat: A @ xhat
    

    xt   = np.zeros((N, n))
    xt[0, T_] = 5.0 # 一開始答案在5而且猴子距離答案有5單位遠
    xt[0, x_] = 5.0
    xhat = np.zeros(n)
    xhat[T_] = 5.0
    xhat[x_] = 5.0
    P    = np.diag([sigma_T2, 1., 1., 1., 1., 1.])

    est = np.zeros((N, n))
    tru = np.zeros((N, n))
    Kh  = np.zeros((N, n))
    Ptr = np.zeros(N)
    Pxx = np.zeros(N)
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
        z = x_meas @ xt[k] + omega
        S = x_meas @ P @ x_meas.T + R
        K = (P @ x_meas.T @ np.linalg.inv(S)).ravel()
        xhat = xhat + K * (z - x_meas @ xhat)
        P = (I - np.outer(K, x_meas.ravel())) @ P

        est[k] = xhat
        tru[k] = xt[k]
        Kh[k] = K
        Ptr[k] = np.trace(P)
        Pxx[k] = P[x_, x_]

        # -------- Kalman 時間更新 (預測) ---------------------------
        xhat = Phi @ xhat + control_u(xhat)
        P = Phi @ P @ Phi.T + Q

    if return_traces:
        return dict(t=np.arange(N)*dt, est=est, tru=tru, Kh=Kh,
                    Ptr=Ptr, Pxx=Pxx, U=Utrace)
    return None

def behavior(kx=kx, seed=3):
    d = simulate(kx=kx, seed=seed)
    tru = d["tru"]
    est = d["est"]
    U = d["U"]
    resid = est[:, x_] - tru[:, x_]
    return dict(true_x_rms=np.sqrt(np.mean(tru[:, x_]**2)),
                resid_rms=np.sqrt(np.mean(resid**2)),
                mean_abs_u=np.mean(np.abs(U[1:])),
                max_abs_u=np.max(np.abs(U[1:])))

# ============================================================== LQR 搜尋
# 結構受限 LQR (只用 x)：控制律 u = kx·x̂ (只有一個純量增益)。
# 標準 Riccati 不適用 (結構約束)，改對 kx 做一維搜尋：
# 對每個 kx 跑閉迴路、算二次成本 J = Σ(x² + Rc·u²)，選 J 最小者。
# 多 seed 平均降低單次雜訊。
def lqr_cost(kx=kx, Rc=Rc, n_seeds=40):
    Js = []
    for s in range(n_seeds):
        d = simulate(kx=kx, seed=100 + s)
        x = d["tru"][:, x_]; u = d["U"]
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e3:
            return np.inf
        Js.append(np.sum(x**2) + Rc * np.sum(u**2))
    return float(np.mean(Js))