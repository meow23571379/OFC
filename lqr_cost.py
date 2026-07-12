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
