"""
kalman_v11_omega_meas.py
====================================================================
會議修正版的 6-state 閉迴路 steering 模型 (v11: 感官雜訊移到量測層)。

相對於 v10 的關鍵改動 (2026-07 與 Bonnie 討論後定案):

  ★ 感官雜訊 ω 的擺法 (最終版, 讀法 A + 自負猴子):
      - 真值 x 保持乾淨的物理量:  x_true = T - H  (不含 ω)
      - 猴子讀到的量測才帶感官雜訊:  z = (T - H) + ω,  ω ~ N(0, σ_ω²)
      - filter 相信自己感官很精準:  R → 0 (極小值), 幾乎全信量測
      → 結果: x̂ ≈ z = (T-H)+ω, 即「估計值 = 乾淨真值 + 感官雜訊」
         (這正是 Bonnie 要的: 真值可畫乾淨線, 估計線帶感官抖動)
      感官雜訊只注入一次 (量測那一刻), 不再在真值層重複加。
      科學宣稱: 猴子客觀感官有雜訊 ω, 但主觀上自負、不校正 (non-Bayesian,
         R→0), 會把每個帶雜訊的讀數當真、追逐雜訊。

  A1'. process noise Q 不再吃世界雜訊 qT:
      T 的 random-walk 擾動是「世界」的事 (上帝視角看得到的物理真值),
      不是猴子感官的雜訊, 概念上分開。Q[x,x] 不再手動繼承 qT。
      T 仍是外生 random-walk 目標; Q[T,T]=qT 保留 (filter 需知 T 會漂移
      才能把 T 當隱藏變數推斷), 但這是「世界漂移」不是「感官雜訊」。

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
from config import *

# # ----------------------------------------------------------------- 參數
# fs = 10.6
# dt = 1.0 / fs
# N  = 220
# g, k_s, b = 6.0, 6.0, 3.2
# T_, H_, x_, r_, rp_, rpp_ = range(6)
# n = 6

# # ----------------------------------------------------------------- 被動動態 Φ
# # 關鍵改動 (A2): 第 6 列 (r'') 不再含 g。
# # 舊: [0,0, g, -k_s, -b, 0]  ->  新: [0,0, 0, -k_s, -b, 0]
# # g 這個「猴子把感知誤差轉成加速度命令」的環節，移到控制輸入 u 裡。
# Phi = np.array([
#     [1,  0, 0,  0,   0,  0],   # T: random walk (外生世界)
#     [0,  1, 0, dt,   0,  0],   # H += dt·r
#     [1, -1, 0,  0,   0,  0],   # x = T - H  (稍後再疊加感官雜訊 ω)
#     [0,  0, 0,  1,  dt,  0],   # r += dt·r'
#     [0,  0, 0,  0,   1, dt],   # r' += dt·r''
#     [0,  0, 0,-k_s, -b,  0],   # r'' = -k_s·r - b·r'  (+ 控制 u，見下)
# ], float)

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

# # ----------------------------------------------------------------- 雜訊設定
# # (1) 世界擾動: T 是 random walk，方差 σ_T² · dt (與舊版一致)
# sigma_T2 = 5.68            # deg²  (per unit time)
# qT = sigma_T2 * dt         # 每步 T 的 process variance

# # (2) 感官雜訊 ω: 只作用在 x，代表猴子感官的 steering-error 雜訊
# sigma_x = 0.30             # deg   (感官雜訊標準差；baseline，可掃描)
# qx = sigma_x**2            # deg²

# # v11 改動: 感官雜訊 ω 已移到量測層，不再進 Q。
# # Q[x,x] 不再手動繼承 qT/qx；T 的擾動是世界的事，會透過真值 x=T-H
# # 自然反映，但那不是「猴子感官的 noise」。Q[T,T]=qT 保留 (filter 需
# # 知 T 會漂移才能把 T 當隱藏變數推斷)，但這是「世界漂移」不是感官雜訊。
# Q = np.zeros((n, n))
# Q[T_, T_]   = qT          # 世界漂移 (filter 需知 T 會動)，非感官雜訊
# Q[x_, x_]   = 1e-6        # 極小 regularizer (x 不再手動繼承任何雜訊)
# Q[rpp_, rpp_] = 1e-4       # 極小 regularizer，避免奇異

# # ----------------------------------------------------------------- 量測
# # A1: 猴子只「看」steering error x（帶感官雜訊）。
# # 量測方程直接量 x（H 不再是被量測的 sensor）。
# # v11: 感官雜訊 ω 在量測那一刻注入 (z = x_true + ω, σ_ω² = qx)。
# # R (filter 內部信念) → 極小: 自負猴子以為感官完美，K_x→1，x̂ ≈ z。
# # R ≠ 真實 ω 方差，這正是「客觀有雜訊但主觀不知」的 non-Bayesian 宣稱。
# H_meas = np.array([[0, 0, 1, 0, 0, 0]], float)   # 只量 x
# sigma_omega2 = qx                                 # 真實感官雜訊方差 (進量測)
# R = np.array([[1e-1]])                             # filter 信念: 自負猴子 R→0

# I = np.eye(n)

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


def main():
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

    np.savez("./lqr_kx_sweep.npz",
             kx=kx_grid, J=Jvals, kx_star=kx_star, Rc=Rc)
    print("\n[saved] lqr_kx_sweep.npz")

if __name__ == "__main__":
    main()