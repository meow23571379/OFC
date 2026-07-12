
from config import *
import numpy as np
from behavior import *

def main():
    # 先重現 v7 佔位版 (kx=g=6) 當健全性檢查

    b6 = behavior()
    print("=== 健全性檢查: kx=6 (佔位版, 應 = v7 的 2.597) ===")
    print("  真實 x RMS :", round(b6["true_x_rms"], 3),
          "| mean|u| :", round(b6["mean_abs_u"], 3))

    
    kx_grid = np.linspace(0.2, 20.0, 100)
    Jvals = np.array([lqr_cost() for kx in kx_grid])
    best_i = int(np.argmin(Jvals))
    kx_star = kx_grid[best_i]

    print(f"\n=== 結構受限 LQR (只用 x), R_cost = {Rc} ===")
    print(f"最佳 kx* = {kx_star:.3f}  (成本 J = {Jvals[best_i]:.1f})")
    print(f"對照 kx=6 (佔位版) 成本 J = {lqr_cost(6.0, Rc):.1f}")
    print("\n成本曲線 (抽樣):")
    for kx in [0.5, 1, 2, 4, 6, 8, 12, kx_star]:
        print(f"  kx = {kx:6.3f} -> J = {lqr_cost(kx, Rc):12.1f}")

    print("\n--- 最佳 kx* 行為 vs 佔位版 kx=6 ---")
    bs = behavior(kx_star, n)
    print(f"  kx*={kx_star:.2f}:  真實 x RMS = {bs['true_x_rms']:.3f} deg, "
          f"mean|u| = {bs['mean_abs_u']:.3f}")
    print(f"  kx =6.00 :  真實 x RMS = {b6['true_x_rms']:.3f} deg, "
          f"mean|u| = {b6['mean_abs_u']:.3f}")

    np.savez("./lqr_kx_sweep_from_mainFile.npz",
             kx=kx_grid, J=Jvals, kx_star=kx_star, Rc=Rc)
    print("\n[saved] lqr_kx_sweep.npz")

if __name__ == "__main__":
    main()