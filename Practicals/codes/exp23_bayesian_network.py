"""Experiment 23 (CO4): Bayesian Network Construction and Visualization
Libraries: pgmpy, NetworkX, Matplotlib
Builds the Burglary-Alarm network. pgmpy is used when installed; otherwise the built-in SimpleBN (identical CPTs) is used."""
import numpy as np, networkx as nx
import matplotlib.pyplot as plt
from common import out_dir, savefig, alarm_network

OUT = out_dir(__file__)
bn = alarm_network()
print("Nodes:", bn.order); print("Edges:", bn.edges())
G = nx.DiGraph(bn.edges()); print("Is DAG:", nx.is_directed_acyclic_graph(G), "| topological order:", list(nx.topological_sort(G)))
print("Markov blanket of Alarm: parents", sorted(G.predecessors("Alarm")), "children", sorted(G.successors("Alarm")))
for n in bn.order:
    parents, table = bn.cpt[n]; print(f"\nCPT P({n}=1 | {', '.join(parents) or 'no parents'}):")
    for k, v in table.items(): print("   ", dict(zip(parents, k)) if parents else "prior", "->", v)
print("\nProbabilities sum to 1 for every row: True (each CPT stores P(X=1|pa); P(X=0|pa)=1-P)")
print("Number of free parameters:", sum(len(bn.cpt[n][1]) for n in bn.order), "vs 2^5 - 1 = 31 for the full joint table")
print("Check: sum over all 32 joint states =", round(sum(bn.joint(dict(zip(bn.order, v))) for v in __import__('itertools').product([0, 1], repeat=5)), 10))

# independence structure via d-separation
try:
    print("\nd-separation: Burglary _|_ Earthquake (marginally):", nx.is_d_separator(G, {"Burglary"}, {"Earthquake"}, set()))
    print("d-separation: Burglary _|_ Earthquake | Alarm:", nx.is_d_separator(G, {"Burglary"}, {"Earthquake"}, {"Alarm"}), "(explaining away)")
    print("d-separation: JohnCalls _|_ MaryCalls | Alarm:", nx.is_d_separator(G, {"JohnCalls"}, {"MaryCalls"}, {"Alarm"}))
except AttributeError:
    pass

# optional pgmpy construction
try:
    from pgmpy.models import DiscreteBayesianNetwork as BNModel
except Exception:
    try: from pgmpy.models import BayesianNetwork as BNModel
    except Exception: BNModel = None
if BNModel is not None:
    from pgmpy.factors.discrete import TabularCPD
    m = BNModel(bn.edges())
    for n in bn.order:
        parents, table = bn.cpt[n]; cols = [(1 - table[k], table[k]) for k in sorted(table, reverse=True)] if parents else [(1 - table[()], table[()])]
        vals = np.array(cols).T.tolist()
        m.add_cpds(TabularCPD(n, 2, vals, evidence=list(parents) or None, evidence_card=[2] * len(parents) or None))
    print("\npgmpy model valid:", m.check_model())
else:
    print("\n(pgmpy not installed -> using built-in SimpleBN; the same code path is used for all numbers above/below)")

layers = {"Burglary": 0, "Earthquake": 0, "Alarm": 1, "JohnCalls": 2, "MaryCalls": 2}
for n, l in layers.items(): G.nodes[n]["layer"] = l
pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
nx.draw(G, pos, ax=ax[0], with_labels=True, node_size=3200, node_color="#9ecae1", edgecolors="k", arrows=True, arrowsize=25, font_size=9)
ax[0].set_title("Bayesian network structure (DAG)")
ax[1].axis("off"); txt = "CPTs  P(X = 1 | parents)\n\n" + "\n".join(f"{n}:  " + "; ".join((f"{dict(zip(bn.cpt[n][0], k))}→{v}" if bn.cpt[n][0] else f"prior={v}") for k, v in bn.cpt[n][1].items()) for n in bn.order)
ax[1].text(0, .95, txt, va="top", family="monospace", fontsize=9, wrap=True); ax[1].set_title("Conditional probability tables")
savefig(fig, OUT, "bayesian_network.png")
