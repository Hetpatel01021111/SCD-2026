#!/usr/bin/env python3
"""Build transparent judge-facing results from historical and layered runs.

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
    with (out / "comparative_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    lines = ["# Judge-ready comparative results", "",
             "The best rows below are post-hoc summaries across completed runs. They are shown with their source, seed and defense so the selection is auditable. The mean ± sample standard deviation across final layered seeds is the primary comparison.", ""]
    for attack in ("label", "backdoor"):
        rs = [r for r in rows if r["attack"] == attack]
        best_acc = max(rs, key=lambda r: r["corrected"])
        if attack == "backdoor":
            best_security = min([r for r in rs if r["asr_after"] != ""], key=lambda r: r["asr_after"])
        else: best_security = None
        final = [r for r in rs if r["source"] == "layered"]
        lines += [f"## {attack.title()}", "", "| View | Baseline | Poisoned | Corrected | Defense | Seed | ASR after |", "|---|---:|---:|---:|---|---:|---:|"]
        for label, r in [("Best corrected accuracy", best_acc), *([("Best ASR", best_security)] if best_security else [])]:
            lines.append(f"| {label} | {r['baseline']:.2f}% | {r['poisoned']:.2f}% | {r['corrected']:.2f}% | {r['defense']} | {r['seed']} | {r['asr_after'] if r['asr_after'] != '' else '—'} |")
        if final:
            for key, label in [("baseline", "Layered mean baseline"), ("poisoned", "Layered mean poisoned"), ("corrected", "Layered mean corrected")]:
                vals = [r[key] for r in final]
                lines.append(f"| {label} | {statistics.mean(vals):.2f}% ± {statistics.stdev(vals):.2f} | — | — | — | 42/43/44 | —")
        lines.append("")
    lines += ["## Interpretation", "", "Best observed values are post-hoc descriptive summaries, not independent test-set selections. The raw JSON logs remain the audit source. A high normal accuracy does not demonstrate backdoor removal; ASR is the security metric. The historical known-trigger result assumes the trigger is known and must not be presented as a generic detector result."]
    (out / "JUDGE_RESULTS.md").write_text("\n".join(lines) + "\n")

    for attack in ("label", "backdoor"):
        rs = [r for r in rows if r["attack"] == attack]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(rs)); width = .25
        for offset, key, label, color in [(-width, "baseline", "Baseline", "#62a956"), (0, "poisoned", "Poisoned", "#ed5b43"), (width, "corrected", "Corrected", "#4f8fe8")]:
            ax.bar([i + offset for i in x], [r[key] for r in rs], width, label=label, color=color)
        ax.set_xticks(list(x), [f"{r['source']}\nseed {r['seed']}" for r in rs], rotation=30, ha="right")
        ax.set_ylim(0, 100); ax.set_ylabel("Normal test accuracy (%)")
        ax.set_title(f"{attack.title()}: all completed runs")
        ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / f"{attack}_all_runs.png", dpi=180); plt.close(fig)


if __name__ == "__main__": main()
