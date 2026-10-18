"""Reproduces every numerical value and figure of the k-house SEIRW manuscript.

Requires numpy and matplotlib.   Run:  python reproduce_results.py
Figures are written to ./figures/ ; all reported numbers are printed.

Model:  K(alpha) = (B + H M_alpha^{-1} D_xi) P,   M_alpha = D_omega + alpha*Q,
        Q = D_G - G   (g_ij = movement from house j to house i),
        R0(alpha) = rho(K(alpha)),  R_E(alpha) = rho(H M_alpha^{-1} D_xi P),  R_D = rho(B P).
"""
import os
import numpy as np
from numpy.linalg import inv, eigvals
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def rho(A):                     return np.max(np.real(eigvals(A)))
def Qof(G):                     return np.diag(G.sum(axis=0)) - G
def K(a, B, H, X, P, Dw, G):    return (B + H @ inv(Dw + a * Qof(G)) @ X) @ P
def R0(a, *p):                  return rho(K(a, *p))
def RE(a, B, H, X, P, Dw, G):   return rho(H @ inv(Dw + a * Qof(G)) @ X @ P)

def perron(Kmat):
    w, V = np.linalg.eig(Kmat); v = V[:, w.real.argmax()].real
    w2, U = np.linalg.eig(Kmat.T); u = U[:, w2.real.argmax()].real
    if v.sum() < 0: v, u = -v, -u
    return u / (u @ v), v       # normalised u^T v = 1, u, v >= 0

def potentials(a, B, H, X, P, Dw, G):
    """Phi = M^{-T} H u,  Psi = M^{-1} X P v   (Theorem 4.2)."""
    u, v = perron(K(a, B, H, X, P, Dw, G)); Minv = inv(Dw + a * Qof(G))
    return Minv.T @ H @ u, Minv @ X @ P @ v

def dR0_dalpha(a, B, H, X, P, Dw, G):
    Phi, Psi = potentials(a, B, H, X, P, Dw, G); k = len(Phi)
    return sum(G[i, j] * Psi[j] * (Phi[i] - Phi[j]) for i in range(k) for j in range(k) if i != j)

def edge_sensitivity(a, B, H, X, P, Dw, G):
    """dR0/dg_ij = alpha * Psi_j (Phi_i - Phi_j)."""
    Phi, Psi = potentials(a, B, H, X, P, Dw, G); k = len(Phi)
    return a * np.array([[Psi[j] * (Phi[i] - Phi[j]) if i != j else 0.0 for j in range(k)] for i in range(k)])

def stationary(G):
    w, V = np.linalg.eig(Qof(G)); pi = V[:, np.abs(w).argmin()].real; return pi / pi.sum()

# ======================= two-house laboratory =======================
Dw2 = np.diag([2., 1.]); G2 = np.array([[0., 1.], [3., 0.]]); H2 = np.diag([1., 3.])
X2 = np.diag([3., 1.]);  P2 = np.diag([2., 1.]); B2 = np.array([[0., 2.], [0., 4.]])
two = (B2, H2, X2, P2, Dw2, G2)
print("=== two-house laboratory ===")
print("R_D =", rho(B2 @ P2))
for a in (0, 1, 5, 20):
    print(f"alpha={a:>2}: R_E = {RE(a, *two):.6f}   R_0 = {R0(a, *two):.6f}")
print("closed form R0(1) =", (55 + np.sqrt(4201)) / 14, "  R0'(0) =", dR0_dalpha(0.0, *two), "(expected 21/2)")

# ======================= five-house configuration =======================
rng = np.random.default_rng(20260913); k = 5
mu = np.linspace(.08, .16, k); sig = np.linspace(.65, .95, k); r = np.linspace(.38, .72, k)
beta = np.linspace(.25, .55, k); eta = np.linspace(.7, 1.4, k); q = np.linspace(.8, 1.3, k)
xi = np.linspace(.7, 1.2, k); omega = np.linspace(.45, .85, k)
C = np.full((k, k), .025); np.fill_diagonal(C, 0)
B5 = np.diag(beta) + C; H5 = np.diag(eta * q); X5 = np.diag(xi)
P5 = np.diag((sig / r) * (1 / (sig + mu))); Dw5 = np.diag(omega)
G5 = np.zeros((k, k))
for j in range(k):
    for i in range(k):
        if i != j: G5[i, j] = .02 + .08 * rng.random()
five = (B5, H5, X5, P5, Dw5, G5)
print("\n=== five-house configuration ===")
print(f"R0(0) = {R0(0, *five):.4f}  R0(20) = {R0(20, *five):.4f}  R_E(0) = {RE(0, *five):.4f}  "
      f"R_E(20) = {RE(20, *five):.4f}  R_D = {rho(B5 @ P5):.4f}")
d0 = dR0_dalpha(0.0, *five); print(f"R0'(0) from Theorem 4.2 = {d0:.7f}")
S = edge_sensitivity(1.0, *five); off = S[~np.eye(k, dtype=bool)]
print("edges at alpha=1: positive", int((off > 0).sum()), " negative", int((off < 0).sum()))
pi = stationary(G5); one = np.ones(k)
REinf = (one @ X5 @ P5 @ H5 @ pi) / (one @ Dw5 @ pi)
print("pi =", np.round(pi, 4), "  R_E(inf) =", round(REinf, 4))

# ======================= four-house configuration =======================
G4 = np.array([[0., .18676055, 1.13409212, .81236097], [.98523368, 0., .81327833, .50700571],
               [.59144856, 1.02896409, 0., .40809778], [.12089721, 1.40339853, 1.43800821, 0.]])
Dw4 = np.diag([.62478676, 1.44991748, 1.99386879, 1.20422586])
H4 = np.diag([.53669501, .40102555, 1.36073196, 2.91421398])
X4 = np.diag([1.12802281, .71707664, 1.89141466, .34346463])
P4 = np.diag([1.33294022, 1.05874074, 1.8659112, 1.294221])
B4 = np.array([[1.07918447, .13585585, .20192385, .44449085], [.37123652, 1.18721717, .05377471, .13350669],
               [.05072876, .35474165, .67764017, .29408311], [.15211338, .29210047, .08140161, 1.05927646]])
four = (B4, H4, X4, P4, Dw4, G4)
grid = np.linspace(0, 20, 401); v4 = np.array([R0(a, *four) for a in grid])
print("\n=== four-house configuration ===")
print(f"R0(0) = {v4[0]:.4f}  min = {v4.min():.4f} at alpha = {grid[v4.argmin()]:.2f}  R0(20) = {v4[-1]:.4f}")

# ============================ figures ============================
# Fig 1: forward-difference error for R0'(0), five-house
hs = 10.0 ** np.arange(-6, 0)
err = [abs((R0(h, *five) - R0(0, *five)) / h - d0) for h in hs]
plt.figure(figsize=(6.4, 4.8)); plt.loglog(hs, err, "o-"); plt.grid(True, which="both", alpha=.3)
plt.xlabel("finite-difference step $h$"); plt.ylabel(r"$|[R_0(h)-R_0(0)]/h-R_0'(0)|$")
plt.title("Validation of the edge-gradient derivative"); plt.tight_layout()
plt.savefig("figures/fig1_derivative_validation.png", dpi=220); plt.close()
# Fig 2: edge sensitivities at alpha = 1
plt.figure(figsize=(6.0, 5.0)); im = plt.imshow(S, cmap="viridis")
plt.colorbar(im, label=r"$\partial R_0/\partial g_{ij}$"); plt.xlabel("source $j$"); plt.ylabel("destination $i$")
plt.xticks(range(k)); plt.yticks(range(k)); plt.title(r"Directed-edge sensitivities at $\alpha=1$")
plt.tight_layout(); plt.savefig("figures/fig2_edge_sensitivities.png", dpi=220); plt.close()
# Fig 3: R0, R_E, R_D versus alpha, five-house
al = np.linspace(0, 20, 401)
plt.figure(figsize=(6.4, 4.8))
plt.plot(al, [R0(a, *five) for a in al], label=r"$R_0(\alpha)$")
plt.plot(al, [RE(a, *five) for a in al], label=r"$R_E(\alpha)$")
plt.axhline(rho(B5 @ P5), ls="--", label=r"$R_D$"); plt.xlabel(r"movement intensity $\alpha$")
plt.ylabel("spectral reproduction number"); plt.title("Reproduction numbers across movement intensity")
plt.xlim(0, 20); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig("figures/fig3_movement_curve.png", dpi=220); plt.close()
# Fig 4: convergence of M_alpha^{-1} to the strong-mixing limit
lim = np.outer(pi, one) / (one @ Dw5 @ pi); als = np.logspace(0, 5, 60)
e4 = [np.linalg.norm(inv(Dw5 + a * Qof(G5)) - lim, 2) for a in als]
plt.figure(figsize=(6.4, 4.8)); plt.loglog(als, e4); plt.grid(True, which="both", alpha=.3)
plt.xlabel(r"$\alpha$"); plt.ylabel(r"$\|M_\alpha^{-1}-M_\infty^{-1}\|_2$")
plt.title("Convergence to the strong-mixing limit"); plt.tight_layout()
plt.savefig("figures/fig4_strong_mixing.png", dpi=220); plt.close()
# Fig 5: two-house laboratory
R0_2 = np.array([R0(a, *two) for a in al]); fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(al, R0_2, lw=2, label=r"$R_0(\alpha)$"); ax[0].plot(al, [3] * len(al), "--", lw=2, label=r"$R_E(\alpha)=3$")
ax[0].plot(al, [4] * len(al), ":", lw=2, label=r"$R_D=4$"); ax[0].set(xlabel=r"Environmental movement intensity $\alpha$",
    ylabel="Spectral radius", title="Reproduction numbers"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(al, R0_2 - 7, lw=2, label=r"$S(\alpha)=R_0(\alpha)-R_D-R_E(\alpha)$"); ax[1].axhline(0, ls="--", lw=.8)
ax[1].set(xlabel=r"Environmental movement intensity $\alpha$", ylabel=r"$S(\alpha)$", title="Mixed-pathway spectral excess")
ax[1].legend(); ax[1].grid(alpha=.3); fig.suptitle("Two-house analytical laboratory"); fig.tight_layout()
fig.savefig("figures/fig5_two_house_components.png", dpi=220); plt.close(fig)
# Fig 6: three response regimes
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(al, R0_2, lw=2, label="Increasing (2-house)"); ax[0].set(xlabel=r"Movement intensity $\alpha$",
    ylabel=r"Full reproduction number $R_0(\alpha)$", title="Increasing response"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(al, [R0(a, *five) for a in al], lw=2, label="Decreasing (5-house)")
ax[1].plot(grid, v4, lw=2, label="Nonmonotone (4-house)"); am = grid[v4.argmin()]
ax[1].annotate(rf"Minimum: $\alpha\approx{am:.2f}$, $R_0\approx{v4.min():.4f}$", (am, v4.min()),
               (4, v4.min() - .06), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax[1].set(xlabel=r"Movement intensity $\alpha$", ylabel=r"Full reproduction number $R_0(\alpha)$",
          title="Decreasing and nonmonotone responses"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("Three movement-response regimes"); fig.tight_layout()
fig.savefig("figures/fig6_movement_regimes.png", dpi=220); plt.close(fig)
print("\nFigures written to ./figures/")