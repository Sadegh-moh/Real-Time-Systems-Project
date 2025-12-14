# phase2/plots.py
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def plot_compare_qos_energy(method_results: Dict[str, Dict[str, Any]]) -> Dict[str, plt.Figure]:
    """
    Creates:
      - QoS comparison (overall + per core)
      - Energy comparison (total + per core)
    """
    methods = list(method_results.keys())
    cores = ["big", "little"]

    # Overall QoS = total_met / total_completed across both cores
    overall_qos = []
    total_energy = []
    for m in methods:
        met = sum(method_results[m]["metrics"][c]["met"] for c in cores)
        comp = sum(method_results[m]["metrics"][c]["completed"] for c in cores)
        qos = (met / comp) if comp > 0 else 0.0
        overall_qos.append(qos)
        total_energy.append(sum(method_results[m]["metrics"][c]["energy"] for c in cores))

    # Figure 1: Overall QoS
    fig_qos = plt.figure()
    plt.bar(np.arange(len(methods)), overall_qos)
    plt.xticks(np.arange(len(methods)), methods)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Overall QoS (deadline-met ratio)")
    plt.title("Overall QoS comparison")

    # Figure 2: Total energy
    fig_energy = plt.figure()
    plt.bar(np.arange(len(methods)), total_energy)
    plt.xticks(np.arange(len(methods)), methods)
    plt.ylabel("Total energy (arbitrary units)")
    plt.title("Total energy comparison")

    # Figure 3: QoS per core (grouped bars)
    fig_qos_core = plt.figure()
    x = np.arange(len(cores))
    w = 0.25
    for i, m in enumerate(methods):
        vals = [method_results[m]["metrics"][c]["qos"] for c in cores]
        plt.bar(x + (i - 1) * w, vals, width=w, label=m)
    plt.xticks(x, cores)
    plt.ylim(0.0, 1.0)
    plt.ylabel("QoS per core")
    plt.title("QoS per core (comparison)")
    plt.legend()

    # Figure 4: Energy per core (grouped bars)
    fig_energy_core = plt.figure()
    x = np.arange(len(cores))
    w = 0.25
    for i, m in enumerate(methods):
        vals = [method_results[m]["metrics"][c]["energy"] for c in cores]
        plt.bar(x + (i - 1) * w, vals, width=w, label=m)
    plt.xticks(x, cores)
    plt.ylabel("Energy per core")
    plt.title("Energy per core (comparison)")
    plt.legend()

    return {
        "compare_overall_qos": fig_qos,
        "compare_total_energy": fig_energy,
        "compare_qos_per_core": fig_qos_core,
        "compare_energy_per_core": fig_energy_core,
    }


def plot_feature_distributions(feature_rows: List[dict], method_name: str) -> Dict[str, plt.Figure]:
    """
    For one method: plot distributions of exec_big, exec_little, slack by assigned core.
    """
    figs: Dict[str, plt.Figure] = {}
    feats = ["exec_big", "exec_little", "slack"]

    # split by core
    rows_big = [r for r in feature_rows if r["assigned_core"] == 0]
    rows_lit = [r for r in feature_rows if r["assigned_core"] == 1]

    for feat in feats:
        big_vals = np.array([r[feat] for r in rows_big], dtype=float)
        lit_vals = np.array([r[feat] for r in rows_lit], dtype=float)

        fig = plt.figure()
        # side-by-side histograms
        bins = 20
        plt.hist(big_vals, bins=bins, alpha=0.7, label="big core")
        plt.hist(lit_vals, bins=bins, alpha=0.7, label="little core")
        plt.xlabel(feat)
        plt.ylabel("count")
        plt.title(f"{method_name}: distribution of {feat} by assigned core")
        plt.legend()

        figs[f"{method_name}_feat_{feat}"] = fig

    return figs
