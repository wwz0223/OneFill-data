"""Pre-specified analysis of listening test v4.3 (written 2026-11-17, before any v4.3 response exists).
Inputs: stim_key.json, items.json, stim_metrics.json (listen_v4_3) and the listeners' result files listen_v4_3_*.json (no practice).
1. Decode: every rating -> (listener, sentence, system) with system in ref / anchor / gslm / ours1 / ours16 / casc.
2. Screening (fixed in advance, same rule as before, scaled to 12 rated sentences): a listener is excluded if the hidden original
   is rated <= 3 in more than 2 sentences, or the zero-fill anchor is rated >= 4 in more than 2 sentences. Organiser and preview runs
   are excluded. The result is reported per listener with the reason.
3. Per system: mean rating, 95% CI from a crossed bootstrap (listeners and sentences resampled independently, 10k, rng 0).
4. Planned comparisons (5): ours1 vs gslm, casc, ours16, anchor, ref. Primary: mean difference with crossed-bootstrap 95% CI and the
   two-sided crossed-bootstrap p (2 x the smaller tail beyond 0), Holm over the 4 tests; the CI and the test then agree because both
   treat listeners AND sentences as random. Secondary (listeners only): Wilcoxon signed-rank on listener means, Holm. A linear mixed model
   rating ~ system + (1 | listener) + (1 | sentence) is the confirmatory model in the protocol; it is fitted separately once data exist.
5. Secondary (RQ3, agreement of metrics with listeners): over the 72 stimuli, Spearman correlation of each metric computed on the exact
   audio heard (stim_metrics.json: UTMOS-g, whole-clip UTMOS, PLCMOS, DNSMOS, DNSMOS-g) with the mean rating of the kept listeners;
   95% CI by a sentence bootstrap (all versions of a sentence resampled together, 2000, rng 0), and the paired difference
   UTMOS-g minus each other metric (same resamples). Repeated on the 4 filled systems only (ref and anchor removed).
usage: python analyze_v4_3.py <listen_v4_3 dir> <results dir> [--simulate N]  -> listen_v4_3_analysis.txt / .json in the results dir"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
SYS = ["ref", "anchor", "gslm", "ours1", "ours16", "casc"]; NAMES = {"ref": "hidden original", "anchor": "zero fill", "gslm": "GSLM-style units", "ours1": "OneFill, 1 step",
       "ours16": "OneFill, 16 steps", "casc": "cascade (ASR+LLM+TTS)"}
D, RD = sys.argv[1], sys.argv[2]; SIM = int(sys.argv[sys.argv.index("--simulate") + 1]) if "--simulate" in sys.argv else 0
K = json.load(open(f"{D}/stim_key.json")); I = json.load(open(f"{D}/items.json")); RATED = [n for n in I["items"] if n not in I["practice"]]

def load():
    runs = []
    if SIM:   # synthetic listeners for testing the pipeline only (never mixed with real files)
        rng = np.random.default_rng(1); base = {"ref": 4.6, "anchor": 1.6, "gslm": 2.4, "ours1": 3.4, "ours16": 3.6, "casc": 3.7}
        for l in range(SIM):
            off = rng.normal(0, .3); tr = []
            for n in RATED:
                so = rng.normal(0, .3); r = {c: int(np.clip(np.rint(base[v] + off + so + rng.normal(0, .6)), 1, 5)) for c, v in K[n].items()}
                tr.append({"key": n, "practice": False, "ratings": r})
            runs.append({"alias": f"SIM{l:02d}", "preview": False, "trials": tr, "_file": "simulated"})
        return runs
    for f in sorted(glob.glob(f"{RD}/listen_v4_3_*.json")):
        r = json.load(open(f, encoding="utf-8"))
        if "trials" not in r: continue   # e.g. listen_v4_3_analysis.json written by this script into the same directory
        r["_file"] = os.path.basename(f); runs.append(r)
    return runs

runs = load(); lines = []; keep = []
for r in runs:
    T = {t["key"]: t for t in r["trials"] if t and not t.get("practice")}
    why = []
    if r.get("preview"): why.append("preview run")
    if set(T) != set(RATED): why.append(f"incomplete ({len(T)}/{len(RATED)} sentences)")
    if not why:
        refl = sum(1 for n in RATED for c, v in K[n].items() if v == "ref" and T[n]["ratings"][c] <= 3)
        anch = sum(1 for n in RATED for c, v in K[n].items() if v == "anchor" and T[n]["ratings"][c] >= 4)
        if refl > 2: why.append(f"hidden original rated <= 3 in {refl} sentences")
        if anch > 2: why.append(f"zero fill rated >= 4 in {anch} sentences")
    lines.append(f"{r['alias']:12s} {r['_file']:40s} {'EXCLUDED: ' + '; '.join(why) if why else 'kept'}")
    if not why: keep.append((r["alias"], T))
L = len(keep); lines.insert(0, f"== screening: {len(runs)} result files, {L} listeners kept")
cmp = [(len(T[n].get("complete", [])), len(T[n]["ratings"])) for _, T in keep for n in RATED if "complete" in T[n]]   # v4.1 logs full plays
if cmp: lines.append(f"   ratings given after hearing the version to the end at least once: {sum(a for a, _ in cmp)}/{sum(b for _, b in cmp)} "
                     f"({100 * sum(a for a, _ in cmp) / max(1, sum(b for _, b in cmp)):.0f}%); rating unlocks at the start of playback")
if L < 2: print("\n".join(lines)); sys.exit("fewer than 2 listeners kept")
Y = np.full((L, len(RATED), len(SYS)), np.nan)   # listener x sentence x system
for i, (_, T) in enumerate(keep):
    for j, n in enumerate(RATED):
        for c, v in K[n].items(): Y[i, j, SYS.index(v)] = T[n]["ratings"][c]
rng = np.random.default_rng(0); B = 10000
li = rng.integers(0, L, (B, L)); si = rng.integers(0, len(RATED), (B, len(RATED)))
def boot(f):   # crossed bootstrap: listeners and sentences resampled independently
    return np.array([f(Y[li[b]][:, si[b]]) for b in range(B)])
lines.append(f"\n== mean naturalness (1-5), {L} listeners x {len(RATED)} sentences, 95% CI crossed bootstrap (listeners and sentences, 10k)")
out = {"n_listeners": L, "systems": {}, "comparisons": []}
for k, s in enumerate(SYS):
    m = float(np.nanmean(Y[:, :, k])); bs = boot(lambda y: np.nanmean(y[:, :, k])); lo, hi = np.percentile(bs, [2.5, 97.5])
    lines.append(f"  {NAMES[s]:24s} {m:.2f} [{lo:.2f}, {hi:.2f}]"); out["systems"][s] = [m, float(lo), float(hi)]
lines.append("\n== planned comparisons: OneFill 1 step minus each other version; crossed bootstrap (primary) and Wilcoxon over listener means (secondary), Holm over 5 each")
res = []
for s in ("gslm", "casc", "ours16", "anchor", "ref"):
    a, b = SYS.index("ours1"), SYS.index(s); lm = np.nanmean(Y[:, :, a], 1) - np.nanmean(Y[:, :, b], 1)
    bs = boot(lambda y: np.nanmean(y[:, :, a]) - np.nanmean(y[:, :, b])); lo, hi = np.percentile(bs, [2.5, 97.5])
    pb = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B))
    pw = float(wilcoxon(lm).pvalue) if np.any(lm != 0) else 1.0
    res.append([s, float(np.nanmean(Y[:, :, a]) - np.nanmean(Y[:, :, b])), float(lo), float(hi), pb, pw])
def holm(ps):
    ps = np.array(ps); order = np.argsort(ps); adj = np.empty_like(ps); run = 0.0
    for rank, i in enumerate(order): run = max(run, min(1.0, ps[i] * (len(ps) - rank))); adj[i] = run
    return adj
hb, hw = holm([r[4] for r in res]), holm([r[5] for r in res])
for r, pa, pw in zip(res, hb, hw):
    lines.append(f"  vs {NAMES[r[0]]:24s} diff {r[1]:+.2f} [{r[2]:+.2f}, {r[3]:+.2f}]  p_Holm(boot)={pa:.3g}{'*' if pa < .05 else ''}   p_Holm(Wilcoxon, listeners)={pw:.3g}")
    out["comparisons"].append({"vs": r[0], "diff": r[1], "ci": [r[2], r[3]], "p_boot": r[4], "p_holm_boot": float(pa), "p_wilcoxon": r[5], "p_holm_wilcoxon": float(pw)})
from scipy.stats import spearmanr
SM = json.load(open(f"{D}/stim_metrics.json")); MET = ["utmos_g", "utmos", "plcmos", "dnsmos_ovrl", "dnsmos_g"]
meanr = {(n, c): float(np.nanmean([T[n]["ratings"][c] for _, T in keep])) for n in RATED for c in K[n]}
for tag2, drop in (("all versions", ()), ("filled systems only (ref, anchor removed)", ("ref", "anchor"))):
    st = [(n, c) for n in RATED for c in K[n] if K[n][c] not in drop]; sent = np.array([n for n, _ in st]); y = np.array([meanr[x] for x in st])
    X = {m: np.array([SM[n][c][m] for n, c in st]) for m in MET}; us = np.unique(sent); rg = np.random.default_rng(0)
    idx = [np.concatenate([np.where(sent == u)[0] for u in rg.choice(us, len(us))]) for _ in range(2000)]
    lines.append(f"\n== metrics vs mean rating, {tag2}, {len(st)} stimuli, Spearman [95% CI, sentence bootstrap 2000]")
    rho = {m: [spearmanr(X[m][i], y[i])[0] for i in idx] for m in MET}; out.setdefault("metric_agreement", {})[tag2] = {}
    for m in MET:
        r0 = spearmanr(X[m], y)[0]; lo, hi = np.nanpercentile(rho[m], [2.5, 97.5]); line = f"  {m:12s} rho {r0:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        if m != "utmos_g":
            d = np.array(rho["utmos_g"]) - np.array(rho[m]); dl, dh = np.nanpercentile(d, [2.5, 97.5]); line += f"   UTMOS-g minus this {np.nanmean(d):+.2f} [{dl:+.2f}, {dh:+.2f}]"
        lines.append(line); out["metric_agreement"][tag2][m] = [float(r0), float(lo), float(hi)]
tag = "_SIMULATED" if SIM else ""
open(f"{RD}/listen_v4_3_analysis{tag}.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
json.dump(out, open(f"{RD}/listen_v4_3_analysis{tag}.json", "w"), indent=1); print("\n".join(lines))
