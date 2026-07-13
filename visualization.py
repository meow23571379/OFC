"""
kalman_v10_rc_spectrum.py
====================================================================
LQR baseline 光譜：把成本函數的省力權重 R_cost 當自由參數，
掃過好幾個數量級，對每個 R_cost 用一維搜尋找最佳回饋增益 kx*。

科學定位 (經 Bonnie 確認)：
  不把文獻的 g=6 當金標準 (兩隻猴子未必代表所有個體)，
  而是把整條「省力權重 R_cost → 最佳 kx*」關係當成一族 baseline 模型。
  之後 signal-dependent noise 的效果 = 在這條光譜上的移動。

畫四條曲線 vs R_cost (log x 軸)：
  (1) 最佳 kx*        —— 省力權重越低，kx 越大 (追越猛)
  (2) 追準度 x RMS    —— 省力權重越低，追越準
  (3) 用力 mean|u|    —— 省力權重越低，用越大力
  (4) 成本組成分解    —— 誤差成本 vs 控制成本 各佔多少
並把文獻 g=6 定位在光譜上 (找出它最接近哪個 R_cost)。

模型本體 import 自 kalman_v10_lqr (= 已驗證正確的 v7 邏輯，kx=6 重現 2.597)。
====================================================================
"""

import numpy as np
import importlib.util
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 註冊中文字體，避免圖上中文變豆腐塊
_cjk = "./NotoSansCJK.ttc"
fm.fontManager.addfont(_cjk)
_cjk_name = fm.FontProperties(fname=_cjk).get_name()
matplotlib.rcParams["font.family"] = _cjk_name
matplotlib.rcParams["axes.unicode_minus"] = False

## ---- import 已驗證的模型 ----
# spec = importlib.util.spec_from_file_location("v10", "./kalman_v10_lqr.py")
# v10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)
# x_ = v10.x_
spec = importlib.util.spec_from_file_location("v11", "./Kalman_v11_omega_meas.py")
v11 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v11)
x_ = v11.x_

def precompute(kx_grid, n_seeds=20):
    """關鍵加速：模擬軌跡與 R_cost 無關，只需算一次。
    對每個 (kx, seed) 快取 Σx²、Σu²、x_rms、mean|u|，之後任何 Rc 只做加權組合。"""
    nK = len(kx_grid)
    sumx2 = np.full(nK, np.inf); sumu2 = np.full(nK, np.inf)
    xrms  = np.full(nK, np.inf); mabu  = np.full(nK, np.inf)
    for j, kx in enumerate(kx_grid):
        xs, us, xr, mu = [], [], [], []
        bad = False
        for s in range(n_seeds):
            d = v11.simulate(kx=kx, seed=200 + s)
            x = d["tru"][:, x_]; u = d["U"]
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e3:
                bad = True; break
            xs.append(np.sum(x**2)); us.append(np.sum(u**2))
            xr.append(np.sqrt(np.mean(x**2))); mu.append(np.mean(np.abs(u[1:])))
        if not bad:
            sumx2[j] = np.mean(xs); sumu2[j] = np.mean(us)
            xrms[j]  = np.mean(xr); mabu[j]  = np.mean(mu)
    return sumx2, sumu2, xrms, mabu


if __name__ == "__main__":
    kx_grid = np.linspace(0.1, 15.0, 80)
    Rc_list = np.logspace(-4, 1, 20)          # 10^-4 ... 10^1，跨 5 個數量級

    print("預先計算所有 kx 的軌跡統計 (與 Rc 無關)...")
    sumx2, sumu2, xrms_all, mabu_all = precompute(kx_grid)

    kx_star, x_rms, mean_u, Jx_part, Ju_part = [], [], [], [], []
    for Rc in Rc_list:
        J = sumx2 + Rc * sumu2            # 對整條 kx_grid 一次算成本
        i = int(np.argmin(J))
        kx_star.append(kx_grid[i])
        Jx_part.append(sumx2[i]); Ju_part.append(Rc * sumu2[i])
        x_rms.append(xrms_all[i]); mean_u.append(mabu_all[i])
        print(f"Rc={Rc:9.2e}  kx*={kx_grid[i]:6.2f}  x_rms={xrms_all[i]:6.3f}  "
              f"mean|u|={mabu_all[i]:7.3f}")

    kx_star = np.array(kx_star); x_rms = np.array(x_rms); mean_u = np.array(mean_u)
    Jx_part = np.array(Jx_part); Ju_part = np.array(Ju_part)

    # 把文獻 g=6 定位在光譜上：找 kx* 最接近 6 的那個 Rc
    idx_g6 = int(np.argmin(np.abs(kx_star - 6.0)))
    Rc_at_g6 = Rc_list[idx_g6]
    print(f"\n文獻 g=6 大約對應 R_cost ≈ {Rc_at_g6:.2e} "
          f"(該處 kx*={kx_star[idx_g6]:.2f})")

    np.savez("rc_spectrum.npz",
             Rc=Rc_list, kx_star=kx_star, x_rms=x_rms, mean_u=mean_u,
             Jx=Jx_part, Ju=Ju_part, Rc_at_g6=Rc_at_g6)

    # ---------------- 畫圖 ----------------
    C_TEAL = "#20808D"; C_RUST = "#A84B2F"; C_GOLD = "#D19900"; C_DARK = "#1B474D"
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.semilogx(Rc_list, kx_star, "o-", color=C_TEAL, lw=2, ms=5)
    a.axhline(6.0, color=C_RUST, ls="--", lw=1.3, label="文獻 g=6")
    a.axvline(Rc_at_g6, color=C_RUST, ls=":", lw=1)
    a.set_xlabel(r"省力權重 $R_{cost}$ (log)"); a.set_ylabel(r"最佳回饋增益 $k_x^*$")
    a.set_title("(1) 最佳增益隨省力權重下降而變猛", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3)

    a = ax[0, 1]
    a.loglog(Rc_list, x_rms, "s-", color=C_DARK, lw=2, ms=5)
    a.axvline(Rc_at_g6, color=C_RUST, ls=":", lw=1, label="g=6 位置")
    a.set_xlabel(r"省力權重 $R_{cost}$ (log)"); a.set_ylabel("追準度：真實 x 的 RMS (deg)")
    a.set_title("(2) 省力權重越低，追得越準", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3, which="both")

    a = ax[1, 0]
    a.loglog(Rc_list, mean_u, "^-", color=C_GOLD, lw=2, ms=5)
    a.axvline(Rc_at_g6, color=C_RUST, ls=":", lw=1, label="g=6 位置")
    a.set_xlabel(r"省力權重 $R_{cost}$ (log)"); a.set_ylabel(r"用力程度：mean $|u|$")
    a.set_title("(3) 省力權重越低，動作越用力", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3, which="both")

    a = ax[1, 1]
    a.plot(x_rms, mean_u, "o-", color=C_TEAL, lw=2, ms=5)
    # 標出 g=6 點
    a.plot(x_rms[idx_g6], mean_u[idx_g6], "*", color=C_RUST, ms=20,
           label=f"文獻 g=6\n(Rc≈{Rc_at_g6:.1e})")
    a.set_xlabel("追準度：真實 x 的 RMS (deg)")
    a.set_ylabel(r"用力程度：mean $|u|$")
    a.set_title("(4) 追準 vs 省力 的 trade-off 曲線", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=.3)

    fig.suptitle("LQR baseline 光譜：省力權重 $R_{cost}$ 掃描 (只用 x 的結構受限控制)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("./rc_spectrum_fig.png", dpi=140, bbox_inches="tight")
    print("\n[saved] rc_spectrum_fig.png")