"""
kalman_v7_omega_x.py
====================================================================
會議修正版 (A1 + A2) 的 6-state 閉迴路 steering 模型。

相對於 v6 / report_figs 的三個結構性改動:

  A1. process noise 改成單一 ω，只作用在猴子「感知到的」steering
      error x 上（讀法 a，選項 1）：
          x_k = (T_k - H_k) + ω_k ,   ω_k ~ N(0, σ_x²)
      也就是「螢幕上的真實偏差 (T - H)」+「猴子感官雜訊 ω」。
      → 神經系統實際拿到的 x 帶有一層感知雜訊。
      其他狀態 (H, r, r', r'') 不再各自注入獨立 process noise。
      T 仍是外生的 random-walk 目標（世界在動），但那是「世界」的
      擾動，不是猴子感官的 ω；兩者概念分開。

  A2. 把 g 從 Φ 第 6 列抽出來，改寫成控制輸入 u：
          state_{k+1} = Φ · state_k + B u_k + (noise)
      u_k 是猴子下給 motor 的命令。目前 u 只是「含 g 的佔位版」，
      維持與舊模型等價的閉迴路行為 (u = g · x̂ 灌進 r'')；
      下一輪會換成 LQR/Riccati 解出的 u = -K x̂。
      → Φ 現在只描述「被動物理演化」，u 才是可被最佳化的主體。

--------------------------------------------------------------------
狀態順序: state = [T, H, x, r, r', r'']
  T   目標位置 (世界座標, random walk)
  H   整合後的 response 朝向   (H += dt · r)
  x   猴子感知到的 steering error = (T - H) + 感官雜訊 ω
  r   response
  r'  角速度
  r'' 角加速度 (由 motor 命令 u 驅動)
單位: 位置類 deg; P 對角 deg²。
====================================================================
"""

import numpy as np

# ----------------------------------------------------------------- 參數
fs = 10.6
dt = 1.0 / fs
N  = 220
g, k_s, b = 6.0, 6.0, 3.2

T_, H_, x_, r_, rp_, rpp_ = range(6)
n = 6

# ----------------------------------------------------------------- 被動動態 Φ
# 關鍵改動 (A2): 第 6 列 (r'') 不再含 g。
# 舊: [0,0, g, -k_s, -b, 0]  ->  新: [0,0, 0, -k_s, -b, 0]
# g 這個「猴子把感知誤差轉成加速度命令」的環節，移到控制輸入 u 裡。
Phi = np.array([
    [1,  0, 0,  0,   0,  0],   # T: random walk (外生世界)
    [0,  1, 0, dt,   0,  0],   # H += dt·r
    [1, -1, 0,  0,   0,  0],   # x = T - H  (稍後再疊加感官雜訊 ω)
    [0,  0, 0,  1,  dt,  0],   # r += dt·r'
    [0,  0, 0,  0,   1, dt],   # r' += dt·r''
    [0,  0, 0,-k_s, -b,  0],   # r'' = -k_s·r - b·r'  (+ 控制 u，見下)
], float)

# ----------------------------------------------------------------- 控制律 A
# 依 Bonnie 的指定: u = A · (estimated state)，A 是 6×6，維度與 Φ 一樣，
# 目前「只有 g 那一格非零」(第 6 列 r''、第 3 行 x)，其餘全部 0。
# 這是佔位版；上 LQR 後整個 A 會被 -K (Riccati 解) 取代，Φ 不用動。
#
#          [0 0 0 0 0 0]
#          [0 0 0 0 0 0]
#   A =    [0 0 0 0 0 0]
#          [0 0 0 0 0 0]
#          [0 0 0 0 0 0]
#          [0 0 g 0 0 0]   <- u 只在 r'' 那格 = g·x
# u = A · x̂，A 只有 (r'', x) 那格 = kx。這裡把回饋增益 kx 參數化，
# 以便 LQR 一維搜尋最佳 kx。佔位版 = g = 6。
def make_A(kx):
    A = np.zeros((n, n)); A[rpp_, x_] = kx
    return A

# ----------------------------------------------------------------- 雜訊設定
# (1) 世界擾動: T 是 random walk，方差 σ_T² · dt (與舊版一致)
sigma_T2 = 5.68            # deg²  (per unit time)
qT = sigma_T2 * dt         # 每步 T 的 process variance

# (2) 感官雜訊 ω: 只作用在 x，代表猴子感官的 steering-error 雜訊
sigma_x = 0.30             # deg   (感官雜訊標準差；baseline，可掃描)
qx = sigma_x**2            # deg²

# ---- 建構 process-noise 協方差 Q (誠實處理相關項) --------------------
# 邏輯: 令世界擾動 w_T 作用在 T；感官雜訊 ω 作用在 x。
# 因為 x = T - H，T 的擾動 w_T 會「繼承」到 x：
#     δT = w_T
#     δx = w_T + ω        (H 這一步不注入自己的 process noise)
# 所以 (T, x) 這兩格不是獨立的，Q 會有 off-diagonal 相關項：
#     Var(T) = qT
#     Var(x) = qT + qx
#     Cov(T, x) = qT
# 其他狀態 (H, r, r', r'') 不注入 process noise (=0)。
# 若只放對角項 (只塞 qx 到 x)，filter 會誤以為 x 與 T 的雜訊獨立，
# 造成系統性估計偏差 —— 這裡明確把相關性算進去。
Q = np.zeros((n, n))
Q[T_, T_]   = qT
Q[x_, x_]   = qT + qx
Q[T_, x_]   = qT
Q[x_, T_]   = qT
Q[rpp_, rpp_] = 1e-4       # 極小 regularizer，避免奇異

# ----------------------------------------------------------------- 量測
# A1: 猴子只「看」steering error x（帶感官雜訊）。
# 量測方程直接量 x（H 不再是被量測的 sensor）。
H_meas = np.array([[0, 0, 1, 0, 0, 0]], float)   # 只量 x
R = np.array([[2.4]])                             # 量測雜訊方差 (deg²)

I = np.eye(n)

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
            # 感官雜訊 ω: 猴子感知到的 x = (真實 T-H) + ω  <<< 選項 1 核心
            omega = np.sqrt(qx) * rng.standard_normal()
            xt[k, x_] = (xt[k, T_] - xt[k, H_]) + omega
            Utrace[k] = u[rpp_]

        # -------- Kalman 量測更新 (量 x) ---------------------------
        z = H_meas @ xt[k] + np.sqrt(R[0, 0]) * rng.standard_normal(1)
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


# ============================================================== LQR 搜尋
# 結構受限 LQR (只用 x)：控制律 u = kx·x̂ (只有一個純量增益)。
# 標準 Riccati 不適用 (結構約束)，改對 kx 做一維搜尋：
# 對每個 kx 跑閉迴路、算二次成本 J = Σ(x² + Rc·u²)，選 J 最小者。
# 多 seed 平均降低單次雜訊。
def lqr_cost(kx, Rc, n_seeds=40):
    Js = []
    for s in range(n_seeds):
        d = simulate(kx=kx, seed=100 + s)
        x = d["tru"][:, x_]; u = d["U"]
        if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e3:
            return np.inf
        Js.append(np.sum(x**2) + Rc * np.sum(u**2))
    return float(np.mean(Js))


def behavior(kx, seed=3):
    d = simulate(kx=kx, seed=seed)
    tru = d["tru"]; est = d["est"]; U = d["U"]
    resid = est[:, x_] - tru[:, x_]
    return dict(true_x_rms=np.sqrt(np.mean(tru[:, x_]**2)),
                resid_rms=np.sqrt(np.mean(resid**2)),
                mean_abs_u=np.mean(np.abs(U[1:])),
                max_abs_u=np.max(np.abs(U[1:])))


if __name__ == "__main__":
    # 先重現 v7 佔位版 (kx=g=6) 當健全性檢查
    b6 = behavior(6.0)
    print("=== 健全性檢查: kx=6 (佔位版, 應 = v7 的 2.597) ===")
    print("  真實 x RMS :", round(b6["true_x_rms"], 3),
          "| mean|u| :", round(b6["mean_abs_u"], 3))

    Rc = 1.0
    kx_grid = np.linspace(0.2, 20.0, 100)
    Jvals = np.array([lqr_cost(kx, Rc) for kx in kx_grid])
    best_i = int(np.argmin(Jvals))
    kx_star = kx_grid[best_i]

    print(f"\n=== 結構受限 LQR (只用 x), R_cost = {Rc} ===")
    print(f"最佳 kx* = {kx_star:.3f}  (成本 J = {Jvals[best_i]:.1f})")
    print(f"對照 kx=6 (佔位版) 成本 J = {lqr_cost(6.0, Rc):.1f}")
    print("\n成本曲線 (抽樣):")
    for kx in [0.5, 1, 2, 4, 6, 8, 12, kx_star]:
        print(f"  kx = {kx:6.3f} -> J = {lqr_cost(kx, Rc):12.1f}")

    print("\n--- 最佳 kx* 行為 vs 佔位版 kx=6 ---")
    bs = behavior(kx_star)
    print(f"  kx*={kx_star:.2f}:  真實 x RMS = {bs['true_x_rms']:.3f} deg, "
          f"mean|u| = {bs['mean_abs_u']:.3f}")
    print(f"  kx =6.00 :  真實 x RMS = {b6['true_x_rms']:.3f} deg, "
          f"mean|u| = {b6['mean_abs_u']:.3f}")

    np.savez("./lqr_kx_sweep_from_v10.npz",
             kx=kx_grid, J=Jvals, kx_star=kx_star, Rc=Rc)
    print("\n[saved] lqr_kx_sweep.npz")
