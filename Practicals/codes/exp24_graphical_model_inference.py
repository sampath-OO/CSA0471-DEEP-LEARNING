"""Experiment 24 (CO4): Graphical Model Sampling and Probabilistic Inference
Libraries: pgmpy, NumPy, Pandas, NetworkX, Matplotlib
Exact inference by enumeration (same result as pgmpy VariableElimination), forward sampling, rejection sampling,
likelihood weighting. pgmpy is used for a cross-check when installed."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import out_dir, savefig, alarm_network

OUT = out_dir(__file__); rng = np.random.default_rng(0)
bn = alarm_network()

# ---- exact inference ---------------------------------------------------------------------------------------
queries = [("Burglary", {}), ("Burglary", {"JohnCalls": 1}), ("Burglary", {"JohnCalls": 1, "MaryCalls": 1}), ("Alarm", {"JohnCalls": 1, "MaryCalls": 1}),
           ("Earthquake", {"JohnCalls": 1, "MaryCalls": 1}), ("Burglary", {"Alarm": 1}), ("Burglary", {"Alarm": 1, "Earthquake": 1}), ("MaryCalls", {"Burglary": 1})]
exact = {}; print("Exact inference by enumeration:")
for t, e in queries:
    p = bn.query(t, e)[1]; exact[(t, str(e))] = p; print(f"  P({t}=1 | {e if e else 'nothing'}) = {p:.5f}")
print("-> Explaining away: P(B | Alarm) = %.4f but P(B | Alarm, Earthquake) = %.4f" % (exact[("Burglary", str({"Alarm": 1}))], exact[("Burglary", str({"Alarm": 1, "Earthquake": 1}))]))

# ---- forward sampling -----------------------------------------------------------------------------------------
N = 100_000; df = pd.DataFrame(bn.forward_sample(N, rng)); print("\nForward sampling (N=%d) marginals:" % N); print(df.mean().round(5).to_string())
print("Exact P(JohnCalls=1) = %.5f, P(MaryCalls=1) = %.5f" % (bn.query("JohnCalls")[1], bn.query("MaryCalls")[1]))
print("\nPandas cross-tab of samples, Alarm vs JohnCalls (counts):\n", pd.crosstab(df.Alarm, df.JohnCalls))

# ---- approximate inference: rejection vs likelihood weighting -----------------------------------------------
target, ev = "Burglary", {"JohnCalls": 1, "MaryCalls": 1}; truth = bn.query(target, ev)[1]
def rejection(n):
    s = df.iloc[:n]; m = np.ones(len(s), bool)
    for k, v in ev.items(): m &= (s[k] == v).values
    return (s[target][m].mean() if m.sum() else np.nan), int(m.sum())
def likelihood_weighting(n):
    num = den = 0.
    for _ in range(n):
        a, w = {}, 1.
        for node in bn.order:
            parents, table = bn.cpt[node]; p1 = table[tuple(a[p] for p in parents)]
            if node in ev: a[node] = ev[node]; w *= p1 if ev[node] == 1 else 1 - p1
            else: a[node] = int(rng.random() < p1)
        num += w * a[target]; den += w
    return num / den
print(f"\nApproximate inference of P({target}=1 | JohnCalls=1, MaryCalls=1); exact = {truth:.5f}")
sizes = [100, 1000, 10000, 100000]; rj, lw = [], []
for n in sizes:
    r, kept = rejection(n); l = likelihood_weighting(min(n, 20000)); rj.append(r); lw.append(l)
    print(f"  N={n:6d}: rejection = {r:.4f} (kept {kept} samples) | likelihood weighting (N={min(n,20000)}) = {l:.4f}")

try:                                                    # pgmpy cross-check
    from pgmpy.models import DiscreteBayesianNetwork as M
except Exception:
    try: from pgmpy.models import BayesianNetwork as M
    except Exception: M = None
if M is not None:
    from pgmpy.factors.discrete import TabularCPD; from pgmpy.inference import VariableElimination
    m = M(bn.edges())
    for n in bn.order:
        pa, tb = bn.cpt[n]; ks = sorted(tb, reverse=True) if pa else [()]
        m.add_cpds(TabularCPD(n, 2, np.array([(1 - tb[k], tb[k]) for k in ks]).T.tolist(), evidence=list(pa) or None, evidence_card=[2] * len(pa) or None))
    print("\npgmpy VariableElimination:", VariableElimination(m).query([target], evidence=ev, show_progress=False).values.round(5))

fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
labels = [f"P({t}|{','.join(k+'='+str(v) for k, v in e.items()) or '∅'})" for t, e in queries]
ax[0].barh(labels, [exact[(t, str(e))] for t, e in queries]); ax[0].set_xscale("log"); ax[0].set_title("Exact posterior probabilities (log scale)"); ax[0].invert_yaxis()
ax[1].semilogx(sizes, rj, "o-", label="rejection sampling"); ax[1].semilogx(sizes, lw, "s-", label="likelihood weighting"); ax[1].axhline(truth, c="k", ls="--", label="exact")
ax[1].set_title("Convergence of approximate inference"); ax[1].set_xlabel("samples"); ax[1].legend(); ax[1].grid()
em = df.mean(); ex = {n: bn.query(n)[1] for n in bn.order}; w = .35; i = np.arange(5)
ax[2].bar(i - w / 2, [ex[n] for n in bn.order], w, label="exact"); ax[2].bar(i + w / 2, [em[n] for n in bn.order], w, label="forward samples"); ax[2].set_xticks(i); ax[2].set_xticklabels(bn.order, rotation=20)
ax[2].set_yscale("log"); ax[2].legend(); ax[2].set_title("Marginals: exact vs. sampled")
savefig(fig, OUT, "bn_inference.png")
df.head(1000).to_csv(f"{OUT}/forward_samples_head.csv", index=False)
