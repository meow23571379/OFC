"""
kalman_visualization.py
====================================================================
統一畫圖程式 — 一次跑出 Kalman steering 專案的所有圖。

★ 只要改開頭這兩行全域設定，就能切換要畫哪個模型版本： ★
      MODEL_FILE  = 要 import 的模型 .py 檔（絕對路徑）
      MODEL_VER   = 版本名稱字串（只用來標在圖上 / 輸出檔名）

  後面所有程式碼一律透過 `current_ver` 這個通用名稱引用模型，
  不寫死任何 "v10" / "v11"，改版只需動上面兩行。

  例：要畫 v11（自負猴子）的圖 →
      MODEL_FILE = ".../kalman_v11_omega_meas.py"
      MODEL_VER  = "v11"

--------------------------------------------------------------------
本檔產生的圖（全部接 current_ver，物理一致）：

  FIG 1  真值 x（乾淨 T−H）vs 估計 $\\hat{x}$ 隨時間        -> fig1_x_true_vs_est_{VER}.png
  FIG 2  重要變數 dashboard（T, H, x, r, u 等）      -> fig2_dashboard_{VER}.png
  FIG 3  共變異數收斂 trace（有 / 無量測更新）        -> fig3_cov_trace_{VER}.png
  FIG 4  Rc 光譜（kx*、追準、用力、trade-off 2×2）    -> fig4_rc_spectrum_{VER}.png

  執行： python kalman_visualization.py
====================================================================
"""

import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==================================================================
# 全域設定 — 改版只動這裡
# ==================================================================
MODEL_FILE = "./kalman_v11_omega_meas.py"
MODEL_VER  = "v11"

OUT_DIR = "."

# ---- 載入模型（統一叫 current_ver）--------------------------------
_spec = importlib.util.spec_from_file_location("current_ver", MODEL_FILE)
current_ver = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(current_ver)

# ---- 從 current_ver 取出狀態索引與常數（不寫死）-------------------
T_, H_, x_, r_, rp_, rpp_ = (current_ver.T_, current_ver.H_, current_ver.x_,
                             current_ver.r_, current_ver.rp_, current_ver.rpp_)
n   = current_ver.n
dt  = current_ver.dt
N   = current_ver.N
G   = current_ver.g            # 佔位 / 文獻增益

# ---- CJK 字型（讓中文標籤正常顯示）-------------------------------
_FONT = "./NotoSansCJK.ttc"
try:
    fm.fontManager.addfont(_FONT)
    _cjk = fm.FontProperties(fname=_FONT).get_name()
    plt.rcParams["font.family"] = _cjk
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass  # 沒字型就用預設，中文可能變豆腐但不會崩

# ---- 統一色盤（design-foundations chart palette）------------------
C_TEAL = "#20808D"   # 主色
C_RUST = "#A84B2F"   # 對照
C_DARK = "#1B474D"   # 深色
C_GOLD = "#D19900"   # 強調
C_GRAY = "#7A7974"

VER = MODEL_VER


def _outp(name):
    return f"{OUT_DIR}/{name}_{VER}.png"


# ==================================================================
# FIG 1 — 真值 x（乾淨 T−H） vs 估計 $\\hat{x}$ 隨時間
# ==================================================================
def fig1_true_vs_est(seed=3):
    d = current_ver.simulate(kx=G, seed=seed)
    t = d["t"]; tru = d["tru"]; est = d["est"]

    fig, ax = plt.subplots(2, 1, figsize=(11, 8),
                           height_ratios=[3, 1.4], sharex=True)

    # 上：真值 x（乾淨）與估計 $\\hat{x}$（帶感官抖動）
    ax[0].plot(t, tru[:, x_], color=C_DARK, lw=2.2,
               label="真值 x  (乾淨 T−H)")
    ax[0].plot(t, est[:, x_], color=C_TEAL, lw=1.3, alpha=0.9,
               label="估計 $\\hat{x}$  (猴子感知，帶 ω)")
    ax[0].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax[0].set_ylabel("steering error x  (deg)")
    ax[0].set_title(f"[{VER}] 真值 x vs 估計 $\\hat{{x}}$ — 估計線 = 乾淨真值 + 感官雜訊 ω",
                    fontsize=13)
    ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3)

    # 下：估計殘差（$\\hat{x}$ − x），大小 ≈ σ_ω
    resid = est[:, x_] - tru[:, x_]
    ax[1].plot(t, resid, color=C_RUST, lw=1.1)
    ax[1].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    rms = np.sqrt(np.mean(resid**2))
    ax[1].set_ylabel("殘差 $\\hat{x}$−x (deg)")
    ax[1].set_xlabel("時間 (s)")
    ax[1].set_title(f"估計殘差 RMS = {rms:.3f} deg  (≈ 感官雜訊 σ_ω)",
                    fontsize=11)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    p = _outp("fig1_x_true_vs_est")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("FIG1 ->", p, f"(殘差RMS={rms:.3f})")
    return p


# ==================================================================
# FIG 2 — 重要變數 dashboard
# ==================================================================
def fig2_dashboard(seed=3):
    d = current_ver.simulate(kx=G, seed=seed)
    t = d["t"]; tru = d["tru"]; est = d["est"]; U = d["U"]

    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # (a) T, H：世界目標與整合 response
    ax[0].plot(t, tru[:, T_], color=C_RUST, lw=2, label="T 目標 (world)")
    ax[0].plot(t, tru[:, H_], color=C_TEAL, lw=2, label="H 整合 response")
    ax[0].set_ylabel("位置 (deg)")
    ax[0].set_title(f"[{VER}] (a) 世界目標 T 與整合 response H", fontsize=12)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    # (b) x 真值 vs 估計
    ax[1].plot(t, tru[:, x_], color=C_DARK, lw=2, label="x 真值 (T−H)")
    ax[1].plot(t, est[:, x_], color=C_GOLD, lw=1.2, label="$\\hat{x}$ 估計")
    ax[1].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax[1].set_ylabel("steering error (deg)")
    ax[1].set_title("(b) steering error：真值 vs 估計", fontsize=12)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    # (c) 控制指令 u 與角速度 r'
    ax[2].plot(t, U, color=C_TEAL, lw=1.6, label="u 控制指令 (打進 r'')")
    ax[2].plot(t, tru[:, rp_], color=C_RUST, lw=1.2, alpha=0.8,
               label="r' 角速度")
    ax[2].axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax[2].set_ylabel("指令 / 角速度")
    ax[2].set_xlabel("時間 (s)")
    ax[2].set_title("(c) 猴子的控制指令 u 與 response 角速度 r'", fontsize=12)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    p = _outp("fig2_dashboard")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    print("FIG2 ->", p)
    return p


# ==================================================================
# FIG 3 — 共變異數收斂 trace（有 / 無量測更新）
# 直接用 current_ver 的 Phi / Q / R / x_meas 傳播「只有協方差」的路徑，
# 協方差與雜訊抽樣無關，可乾淨比較「有量測」vs「純預測」。
# ==================================================================
def _cov_path(use_measurements):
    Phi = current_ver.Phi
    Q   = current_ver.Q
    R   = current_ver.R
    Hm  = current_ver.x_meas
    I   = np.eye(n)
    P = np.diag([current_ver.sigma_T2, 1., 1., 1., 1., 1.])
    tr = np.zeros(N); diag = np.zeros((N, n))
    for k in range(N):
        if use_measurements:
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            P = (I - K @ Hm) @ P
        tr[k] = np.trace(P); diag[k] = np.diag(P)
        P = Phi @ P @ Phi.T + Q          # 時間更新（預測）
    return tr, diag


def fig3_cov_trace():
    t = np.arange(N) * dt
    tr_on,  diag_on  = _cov_path(True)
    tr_off, diag_off = _cov_path(False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    # 左：trace(P) 總不確定度
    ax[0].plot(t, tr_on,  color=C_TEAL, lw=2, label="有量測更新")
    ax[0].plot(t, tr_off, color=C_RUST, lw=2, ls="--", label="純預測 (無量測)")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("時間 (s)"); ax[0].set_ylabel("trace(P)")
    ax[0].set_title(f"[{VER}] 總不確定度 trace(P)：量測把它壓住", fontsize=12)
    ax[0].legend(fontsize=10); ax[0].grid(alpha=0.3, which="both")

    # 右：x 這一格的不確定度 var_xx
    ax[1].plot(t, diag_on[:, x_],  color=C_TEAL, lw=2, label="有量測更新")
    ax[1].plot(t, diag_off[:, x_], color=C_RUST, lw=2, ls="--",
               label="純預測 (無量測)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("時間 (s)"); ax[1].set_ylabel("var$_{xx}$ (deg²)")
    ax[1].set_title("steering error x 的估計不確定度", fontsize=12)
    ax[1].legend(fontsize=10); ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    p = _outp("fig3_cov_trace")
    fig.savefig(p, dpi=135, bbox_inches="tight"); plt.close(fig)
    print("FIG3 ->", p)
    return p


# ==================================================================
# FIG 4 — Rc 光譜（kx*、追準、用力、trade-off）
# 用 current_ver.simulate / lqr_cost。先 precompute 每個 (kx,seed) 的
# Σx²、Σu²、x_rms、mean|u|（與 Rc 無關），再對每個 Rc 選最佳 kx*。
# ==================================================================
def _precompute(kx_grid, n_seeds=20):
    m = len(kx_grid)
    SX = np.zeros((m, n_seeds)); SU = np.zeros((m, n_seeds))
    XR = np.zeros((m, n_seeds)); MU = np.zeros((m, n_seeds))
    for i, kx in enumerate(kx_grid):
        for s in range(n_seeds):
            d = current_ver.simulate(kx=kx, seed=100 + s)
            x = d["tru"][:, x_]; u = d["U"]
            if (not np.all(np.isfinite(x))) or np.max(np.abs(x)) > 1e3:
                SX[i, s] = SU[i, s] = np.inf
                XR[i, s] = MU[i, s] = np.inf
                continue
            SX[i, s] = np.sum(x**2)
            SU[i, s] = np.sum(u**2)
            XR[i, s] = np.sqrt(np.mean(x**2))
            MU[i, s] = np.mean(np.abs(u[1:]))
    return SX, SU, XR, MU


def fig4_rc_spectrum(n_seeds=20, n_kx=60):
    kx_grid = np.linspace(0.05, 6.5, n_kx)
    Rc_grid = np.logspace(-4, 1, 20)

    SX, SU, XR, MU = _precompute(kx_grid, n_seeds=n_seeds)
    meanSX = SX.mean(1); meanSU = SU.mean(1)
    meanXR = XR.mean(1); meanMU = MU.mean(1)

    kx_star = np.zeros(len(Rc_grid))
    xr_star = np.zeros(len(Rc_grid))
    mu_star = np.zeros(len(Rc_grid))
    for j, Rc in enumerate(Rc_grid):
        J = meanSX + Rc * meanSU
        idx = int(np.nanargmin(J))
        kx_star[j] = kx_grid[idx]
        xr_star[j] = meanXR[idx]
        mu_star[j] = meanMU[idx]

    # 文獻 g 落點：找 kx* 最接近 G 的 Rc
    j_g = int(np.argmin(np.abs(kx_star - G)))
    Rc_at_g = Rc_grid[j_g]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"[{VER}] LQR baseline 光譜：省力權重 R_cost 掃描 (只用 x 的結構受限控制)",
                 fontsize=14, y=0.995)

    # (1) kx* vs Rc
    a = ax[0, 0]
    a.plot(Rc_grid, kx_star, "o-", color=C_TEAL, lw=2)
    # a.axhline(G, color=C_RUST, ls="--", lw=1.5, label=f"文獻 g={G:g}")
    a.axvline(Rc_at_g, color=C_RUST, ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("省力權重 R$_{cost}$ (log)"); a.set_ylabel("最佳回饋增益 k$_x^*$")
    a.set_title("(1) 最佳增益隨省力權重下降而變猛", fontsize=11)
    a.legend(fontsize=9, loc="lower left"); a.grid(alpha=0.3, which="both")

    # (2) 追準度 x_rms vs Rc
    a = ax[0, 1]
    a.plot(Rc_grid, xr_star, "s-", color=C_DARK, lw=2)
    a.axvline(Rc_at_g, color=C_RUST, ls=":", lw=1, label="g 位置")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("省力權重 R$_{cost}$ (log)"); a.set_ylabel("追準度：真實 x 的 RMS (deg)")
    a.set_title("(2) 省力權重越低，追得越準", fontsize=11)
    a.legend(fontsize=9, loc="upper left"); a.grid(alpha=0.3, which="both")

    # (3) 用力 mean|u| vs Rc
    a = ax[1, 0]
    a.plot(Rc_grid, mu_star, "^-", color=C_GOLD, lw=2)
    a.axvline(Rc_at_g, color=C_RUST, ls=":", lw=1, label="g 位置")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("省力權重 R$_{cost}$ (log)"); a.set_ylabel("用力程度：mean |u|")
    a.set_title("(3) 省力權重越低，動作越用力", fontsize=11)
    a.legend(fontsize=9, loc="upper right"); a.grid(alpha=0.3, which="both")

    # (4) trade-off 曲線
    a = ax[1, 1]
    a.plot(xr_star, mu_star, "o-", color=C_TEAL, lw=2)
    a.plot(xr_star[j_g], mu_star[j_g], "*", color=C_RUST, ms=22,
           label=f"文獻 g={G:g}\n(Rc≈{Rc_at_g:.1e})")
    a.set_xlabel("追準度：真實 x 的 RMS (deg)"); a.set_ylabel("用力程度：mean |u|")
    a.set_title("(4) 追準 vs 省力 的 trade-off 曲線", fontsize=11)
    a.legend(fontsize=9, loc="upper right"); a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = _outp("fig4_rc_spectrum")
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    # 順手存資料
    np.savez(f"{OUT_DIR}/rc_spectrum_{VER}.npz",
             Rc=Rc_grid, kx_star=kx_star, x_rms=xr_star, mean_u=mu_star,
             Rc_at_g=Rc_at_g)
    print("FIG4 ->", p, f"(g={G:g} @ Rc≈{Rc_at_g:.2e})")
    return p

def fig5_panels(seed=3):
    """
    FIG 5 — Kalman Gain K 時間分布
    ====================================================================
    K 是 6 維向量（每個 state 對應一個 gain），來自單一純量量測 z（只量 x）。
    更新公式: x̂_new = x̂_old + K*(z - H·x̂_old)
    K[i] = P[i,x] / S,  S = P[x,x] + R

    物理意義:
      K[x]  ≈ 1  → x 直接被量，幾乎完全相信量測
      K[T]  ≈ 1  → filter 從 x 推斷 T（因 P[T,x] ≈ P[x,x]，T 與 x 高度相關）
      K[H,r,r',r''] ≈ 0 → 與 x 量測幾乎不相關，幾乎不更新

    Layout:
      Row 1 (full): 所有 6 個 K 疊圖（說明量測如何分配到各 state）
      Row 2: K[x] vs K[T] — 主要 gain pair，穩態均 ≈ 1
      Row 3: K[H], K[r], K[r'], K[r''] — 穩態均 ≈ 0
    ====================================================================
    """
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as mgridspec

    d    = current_ver.simulate(kx=G, seed=seed)
    t    = d["t"]
    Kh   = d["Kh"]   # (N, 6)  — K[i] at each time step

    s_labels = [r"$K_{T}$",  r"$K_{H}$",  r"$K_{x}$",
                r"$K_{r}$",  r"$K_{r'}$", r"$K_{r''}$"]
    s_desc   = ["T (inferred from x)",
                "H (near zero)",
                "x (directly measured)",
                "r (near zero)",
                "r' (near zero)",
                "r'' (near zero)"]
    idx_list = [T_, H_, x_, r_, rp_, rpp_]
    colors6  = [C_TEAL, C_RUST, C_DARK, C_GOLD, C_GRAY, "#9B59B6"]

    fig = plt.figure(figsize=(13, 11))
    outer = mgridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                               top=0.92, bottom=0.07, left=0.09, right=0.97)

    # ---- Row 1: overlay — all 6 K ----
    ax_all = fig.add_subplot(outer[0])
    for i, idx in enumerate(idx_list):
        ax_all.plot(t, Kh[:, idx], color=colors6[i], lw=1.6, label=s_labels[i])
    ax_all.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax_all.set_ylabel("K (gain)", fontsize=11)
    ax_all.set_title(
        f"[{VER}]  (a) Kalman gain K over time — one scalar measurement z = x + \u03c9\n"
        r"$K_i = P_{i,x}\,/\,S$ : how much each state estimate is corrected by the x measurement",
        fontsize=10)
    ax_all.legend(ncol=6, fontsize=9, framealpha=0.9,
                  loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax_all.spines[["top", "right"]].set_visible(False)
    ax_all.grid(lw=0.4, alpha=0.35, color="lightgray")
    ax_all.set_xlim([0, t[-1]])
    ax_all.set_ylim([-0.35, 1.15])

    # ---- Row 2: K[x] and K[T] — the dominant pair ----
    inner2 = mgridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.32)

    pairs = [(x_, r"$K_x$  (directly measured state)",   C_DARK),
             (T_, r"$K_T$  (inferred: $P_{T,x}\approx P_{x,x}$)", C_TEAL)]
    for col, (idx, title, col_c) in enumerate(pairs):
        ax = fig.add_subplot(inner2[col])
        ss = float(Kh[-50:, idx].mean())
        ax.plot(t, Kh[:, idx], color=col_c, lw=1.8)
        ax.axhline(ss, color=col_c, lw=1.0, ls=":", alpha=0.8,
                   label=f"ss = {ss:.4f}")
        ax.axhline(0,  color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"(b) {title}", fontsize=9)
        ax.set_ylabel("K", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.35, color="lightgray")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        ax.set_xlim([0, t[-1]])

    # ---- Row 3: the near-zero gains ----
    inner3 = mgridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.42)
    minor = [(H_,   r"$K_H$",    C_RUST,    "(c)"),
             (r_,   r"$K_r$",    C_GOLD,    "(d)"),
             (rp_,  r"$K_{r'}$", C_GRAY,    "(e)"),
             (rpp_, r"$K_{r''}$","#9B59B6",  "(f)")]
    for col, (idx, lbl, col_c, plbl) in enumerate(minor):
        ax = fig.add_subplot(inner3[col])
        ss = float(Kh[-50:, idx].mean())
        ax.plot(t, Kh[:, idx], color=col_c, lw=1.5)
        ax.axhline(ss, color=col_c, lw=1.0, ls=":", alpha=0.8)
        ax.axhline(0,  color="gray", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"{plbl} {lbl}\nss={ss:.1e}", fontsize=9)
        ax.set_ylabel(lbl, fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(lw=0.4, alpha=0.35, color="lightgray")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, prune="both"))
        ax.set_xlim([0, t[-1]])

    p = _outp("fig5_K_panels")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"FIG5 -> {p}")
    print("  K_i = P[i,x] / S  (single scalar measurement: z = x + omega)")
    for i, idx in enumerate(idx_list):
        ss = float(Kh[-50:, idx].mean())
        print(f"    {s_labels[i]:12s}  ss = {ss:.6f}   ({s_desc[i]})")
    return p


# ==================================================================
# 一次跑出所有圖
# ==================================================================
def run_all():
    print(f"=== 畫圖：模型版本 = {VER}  ({MODEL_FILE}) ===")
    fig1_true_vs_est()
    fig2_dashboard()
    fig3_cov_trace()
    fig4_rc_spectrum()
    fig5_panels()
    print("=== 全部完成 ===")


if __name__ == "__main__":
    run_all()
