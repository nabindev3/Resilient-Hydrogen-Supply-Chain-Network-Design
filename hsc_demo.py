"""
hsc-hybrid-demo - Five-node hydrogen siting MILP with a trained surrogate
embedded directly in Gurobi (Appendix A of the hybrid AI-OR framework report).

The idea in one screen:
  * The solver owns what has to be GUARANTEED - mass balance, capacity, and the
    discrete build decisions.
  * A learned model owns what is nonlinear and messy - here, an operating penalty
    that captures compression / boil-off stress and the fragility you get from
    piling production onto one or two nodes.
  * The seam is `add_predictor_constr`: the trained MLP becomes constraints
    *inside* the exact MILP, so the solver optimises against it directly.

Run:  python hsc_demo.py
Deps: gurobipy, gurobi-machinelearning, scikit-learn, numpy, matplotlib
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from gurobi_ml import add_predictor_constr

# ----------------------------------------------------------------------------
# 1. Problem data - a tiny five-node hydrogen network
# ----------------------------------------------------------------------------
N      = 5                                  # candidate production nodes
DEMAND = 100.0                              # H2 that must be delivered (mass balance)
CAP    = np.full(N, 40.0)                   # per-node nameplate capacity
CAPEX  = np.full(N, 20.0)                   # fixed cost to build a node
UNIT   = np.array([1.00, 1.07, 1.14, 1.21, 1.28])   # per-unit production cost
LAM    = 1.0                                # weight on the learned operating penalty

# The "true" operating penalty we are standing in for. It is nonlinear and
# strictly convex per node, so loading one node hard (compression / boil-off
# stress, single-point fragility) is punished far more than spreading the same
# mass across several. SCALE just puts it in the same units as cost.
SCALE = 0.004449
def true_penalty(x):
    return SCALE * np.sum(np.asarray(x) ** 2.5)

# ----------------------------------------------------------------------------
# 2. Train the surrogate - a small MLP that learns true_penalty over the box
# ----------------------------------------------------------------------------
rng    = np.random.default_rng(0)
X_train = rng.uniform(0.0, CAP[0], size=(12000, N))
y_train = SCALE * (X_train ** 2.5).sum(axis=1)
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                 max_iter=4000, random_state=0),
)
mlp.fit(X_train, y_train)
print(f"surrogate trained: R^2 = {mlp.score(X_train, y_train):.4f}")

# ----------------------------------------------------------------------------
# 3. Build + solve the siting MILP, optionally with the surrogate embedded
# ----------------------------------------------------------------------------
def solve(use_surrogate):
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.MIPGap = 1e-6

    y = m.addMVar(N, vtype=GRB.BINARY)              # build / no-build per node
    x = m.addMVar(N, lb=0.0, ub=CAP)                # production per node
    m.addConstr(x <= CAP * y)                       # capacity (only built nodes run)
    m.addConstr(x.sum() == DEMAND)                  # mass balance - hard guarantee

    cost = CAPEX @ y + UNIT @ x
    if use_surrogate:
        pen = m.addMVar((1, 1), lb=-GRB.INFINITY)
        add_predictor_constr(m, mlp, x.reshape(1, -1), pen)   # <-- the seam
        m.setObjective(cost + LAM * pen[0, 0], GRB.MINIMIZE)
    else:
        m.setObjective(cost, GRB.MINIMIZE)

    m.optimize()
    assert m.Status == GRB.OPTIMAL, f"solver status {m.Status}"
    return x.X.copy(), y.X.copy()

x_cost, y_cost   = solve(use_surrogate=False)   # cost-only baseline
x_hyb,  y_hyb    = solve(use_surrogate=True)    # hybrid: surrogate embedded

# ----------------------------------------------------------------------------
# 4. Report - what changed, and did the guarantees hold
# ----------------------------------------------------------------------------
def show(tag, x):
    built = int((x > 1e-6).sum())
    pen   = true_penalty(x)
    print(f"\n{tag}")
    print(f"  production per node : {np.round(x, 1)}")
    print(f"  nodes built         : {built}")
    print(f"  demand delivered    : {x.sum():.1f} / {DEMAND:.0f}  (mass balance)")
    print(f"  true operating pen.  : {pen:.1f}")
    return built, pen

print("\n" + "=" * 64)
b0, p0 = show("COST-ONLY  (solver alone, penalty ignored)", x_cost)
b1, p1 = show("HYBRID     (trained surrogate embedded in Gurobi)", x_hyb)
print("\n" + "=" * 64)
print(f"nodes: {b0} -> {b1}   |   true penalty: {p0:.0f} -> {p1:.0f} "
      f"({100 * (p0 - p1) / p0:.0f}% lower)   |   demand met exactly in both")
print("=" * 64)

# ----------------------------------------------------------------------------
# 5. Figure A1 - cost-only vs hybrid production profile
# ----------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = np.arange(N)
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.bar(idx - w / 2, x_cost, w, label=f"cost-only  (pen {p0:.0f})", color="#c44e52")
    ax.bar(idx + w / 2, x_hyb,  w, label=f"hybrid      (pen {p1:.0f})", color="#4c72b0")
    ax.axhline(CAP[0], ls="--", lw=0.8, color="0.5")
    ax.text(N - 1, CAP[0] + 0.6, "capacity", ha="right", va="bottom", fontsize=8, color="0.4")
    ax.set_xticks(idx, [f"node {i}" for i in idx])
    ax.set_ylabel("H$_2$ production")
    ax.set_title("Embedding the surrogate spreads production (demand still met exactly)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figure_a1.png", dpi=150)
    print("\nsaved figure_a1.png")
except Exception as e:  # matplotlib is optional
    print(f"\n(figure skipped: {e})")
