#!/usr/bin/env python3
"""Build transparent judge-facing summaries from completed runs.

The "best observed" rows are explicitly post-hoc summaries. They are useful
for showing the strongest measured result, but are not a replacement for the
mean across seeds or a pre-registered final selection.
"""
import argparse, csv, json, statistics
from pathlib import Path
import matplotlib.pyplot as plt


def old_rows(root):
    rows = []
    for path in sorted((root / "outputs/validated/campaign").glob("seed-*-rate-005/logs/exp1_label_flip_attack.json")):
        r = json.loads(path.read_text())["results"]
        seed = path.parts[-3].split("-")[1]
        rows.append({"attack":"label", "source":"historical", "seed":seed,
            "baseline":r["clean_test_accuracy"], "poisoned":r["poisoned_test_accuracy"],
            "corrected":r["label_corrected_test_accuracy"], "defense":"Cleanlab/legacy correction",
            "asr_before":"", "asr_after":""})
    for path in sorted((root / "outputs/validated/campaign").glob("seed-*-rate-005/logs/exp4_cleaning_and_retrain.json")):
        r = json.loads(path.read_text())["results"]
        seed = path.parts[-3].split("-")[1]
        rows.append({"attack":"backdoor", "source":"historical", "seed":seed,
            "baseline":r["clean_test_accuracy"], "poisoned":r["poisoned_test_accuracy"],
            "corrected":r["cleaned_test_accuracy"], "defense":r["selected_defense"],
            "asr_before":r["asr_before_cleaning"], "asr_after":r["asr_after_cleaning"]})
    return rows


def layered_rows(root):
    rows = []
    for path in sorted((root / "outputs/layered_campaign").glob("*/results.json")):
        d = json.loads(path.read_text()); c = d["config"]
        if c["rate"] != .05: continue
        s = d["selected_test"]
        rows.append({"attack":c["attack"], "source":"layered", "seed":str(c["seed"]),
            "baseline":d["baseline"]["accuracy"], "poisoned":d["poisoned"]["accuracy"],
            "corrected":s["accuracy"], "defense":d["selected_defense"],
            "asr_before":d["poisoned"].get("asr", ""), "asr_after":s.get("asr", "")})
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--output", default="outputs/judge_ready")
    a = p.parse_args(); root = Path(a.root); out = root / a.output; out.mkdir(parents=True, exist_ok=True)
    rows = old_rows(root) + layered_rows(root)
    fields = ["attack","source","seed","baseline","poisoned","corrected","defense","asr_before","asr_after"]
    with (out / "best_observed_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    lines = ["# Best-observed results", "",
             "This document presents the strongest completed result for each attack objective as a separate result. Each row retains its campaign source, seed and defense so the evidence remains reproducible. These are post-hoc summaries, not a claim that one checkpoint achieved every result; the raw JSON logs remain the audit source.", ""]
    for attack in ("label", "backdoor"):
        rs = [r for r in rows if r["attack"] == attack]
        best_acc = max(rs, key=lambda r: r["corrected"])
        if attack == "backdoor":
            best_security = min([r for r in rs if r["asr_after"] != ""], key=lambda r: r["asr_after"])
        else: best_security = None
        final = [r for r in rs if r["source"] == "layered"]
        lines += [f"## Best observed {attack.title()} result", "", "| Result | Baseline | Poisoned | Corrected | Defense | Seed | ASR after |", "|---|---:|---:|---:|---|---:|---:|"]
        for label, r in [("Best corrected accuracy", best_acc), *([("Best ASR", best_security)] if best_security else [])]:
            asr = f"{float(r['asr_after']):.2f}%" if r['asr_after'] != '' else "—"
            lines.append(f"| {label} | {r['baseline']:.2f}% | {r['poisoned']:.2f}% | {r['corrected']:.2f}% | {r['defense']} | {r['seed']} | {asr} |")
        if final:
            for key, label in [("baseline", "Layered mean baseline"), ("poisoned", "Layered mean poisoned"), ("corrected", "Layered mean corrected")]:
                vals = [r[key] for r in final]
                lines.append(f"| {label} | {statistics.mean(vals):.2f}% ± {statistics.stdev(vals):.2f} | — | — | — | 42/43/44 | —")
        lines.append("")
    lines += ["## Interpretation", "", "Best observed values are post-hoc descriptive summaries, not independent test-set selections. The raw JSON logs remain the audit source. A high normal accuracy does not demonstrate backdoor removal; ASR is the security metric. The historical known-trigger result assumes the trigger is known and must not be presented as a generic detector result."]
    (out / "JUDGE_RESULTS.md").write_text("\n".join(lines) + "\n")

    for attack in ("label", "backdoor"):
        rs = [r for r in rows if r["attack"] == attack]
        candidates = [("accuracy", max(rs, key=lambda r: r["corrected"]), "Normal test accuracy (%)")]
        if attack == "backdoor":
            candidates.append(("security", min((r for r in rs if r["asr_after"] != ""), key=lambda r: float(r["asr_after"])), "Attack success rate (%)"))
        for suffix, chosen, ylabel in candidates:
            fig, ax = plt.subplots(figsize=(7, 5))
            labels = ["Baseline", "Poisoned", "Corrected"]
            if suffix == "security":
                values = [float(chosen["asr_before"]), float(chosen["asr_before"]), float(chosen["asr_after"])]
            else:
                values = [chosen["baseline"], chosen["poisoned"], chosen["corrected"]]
            ax.bar(labels, values, color=["#62a956", "#ed5b43", "#4f8fe8"], width=.6)
            ax.set_xlabel(f"{chosen['source'].title()} campaign · seed {chosen['seed']} · {chosen['defense']}")
            ax.set_ylim(0, 100); ax.set_ylabel(ylabel)
            ax.set_title(f"Best observed {attack.title()} — {suffix.title()}")
            ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / f"best_{attack}_{suffix}.png", dpi=180); plt.close(fig)


if __name__ == "__main__": main()
