"""Sets the exact formula of every WikiText-2 experiment on the nb site from its title (titles are unchanged).
    python scripts/nb_formulas.py [--dry]
Notation of the notes: I_t = (1-t) x_0 + t x_1, exponential clock (kappa = sigma), jump Phi_sigma(x) = (1-sigma) x + sigma delta_sigma(x),
tau = decoding-error clock. Every flow map minimises L_diag + L_off with L_off = - target . log delta_sigma(x), target stopped."""
import re
import subprocess
import sys

NB = ".venv/bin/nb"
TGT = {
    "semigroup": r"$\bar\delta = \gamma\hat\delta_{\sigma_1}(x) + (1-\gamma)\hat\delta_{\sigma_2}(\hat\Phi_{\sigma_1}(x))$, $\gamma = \frac{\sigma_1(1-\sigma_2)}{\sigma_3}$",
    "lagrangian": r"$\bar\delta = \bar D(\hat\Phi_\sigma(x)) - (1-\sigma)\sigma\,\partial_\sigma\hat\delta_\sigma(x)$",
    "eulerian": r"$\bar\delta = (1-\sigma)\bar D(x) + \sigma\hat\delta_\sigma(x) + \sigma\nabla\hat\delta_\sigma(x)(\bar D(x)-x) - \sigma(1-\sigma)\partial_\sigma\hat\delta_\sigma(x)$",
}
DUR = {"semigroup": r"$\sigma_3$ with density $2\sigma_3$, $\sigma_1 \sim \mathcal{U}[0,\sigma_3]$", "other": r"$\sigma \sim \mathcal{U}[0,1]$"}
PAIRS = r"$h \sim \mathcal{U}[0,1]$, $\tau_s \sim \mathcal{U}[0,1-h]$, $\tau_e = \tau_s + h$, or $(0,1)$ with probability $\frac{1}{32}$; $\sigma = \frac{t_e - t_s}{1 - t_s}$"
RANDOM_C = r"$x^l = x_1^l$ on $C$, $P(l \in C) = f \sim \mathcal{U}[0,1]$, share \cfg{loss.context_frac} of samples; losses over $l \notin C$"
MODEL_C = r"$x^l = x_1^l$ on $C = \{l : \max_i p^{l,i} > h\}$, $p = \mathrm{sg}[\hat\delta_1(I_t)]$ ($\hat D(I_t)$ for a denoiser), $h$ = \cfg{loss.adaptive_h}, share \cfg{loss.adaptive_frac} of samples; losses over $l \notin C$"
RIL = r"from step \cfg{loss.ril_start}, 4 commit rounds $p_k = \hat\delta_1(x_{(k-1)})$: $L_{ril} = -\sum_k \sum_{l \notin C_k} x_1^l \cdot \log p_k^l$"
SD = r"$L_{sd}$: same with target $\mathrm{sg}[q^l]$, $q$ a frozen copy at the last round"
DENOISER = r"$L = \mathbb{E}[-x_1 \cdot \log \hat D(I_t)]$, $I_t = (1-t)x_0 + t x_1$, $\tau(t) \sim \mathcal{U}[0,1]$, weight 1, network told: \cfg{model.cond}"
BECK = r"$L = \lVert T(x) - \mathrm{sg}[T(x) + \nabla T(x)(x_1 - x_0)] \rVert^2 - \sum_{l: t^l > 0.75} x_1^l \cdot \log T(x)^l - x_1 \cdot \log T(x_1)$, $t \sim \mathcal{U}[0,1]$, clean positions with probability $f \sim \mathcal{U}[0,1]$, optimal-transport pairing, weights $(\ell+1)^{-1/2}$"


def formula(title):
    t = title.lower()
    if "beckmann" in t:
        if "refinement" in t:
            return BECK[:BECK.index(", $t")] + r"; + from step \cfg{dbtm.ril_start}: $-\sum_k \sum_{l \notin C_k} (x_1^l + \mathrm{sg}[q^l]) \cdot \log T(x_{(k)})^l$ over 4 commit rounds (data tokens, threshold 0.9; $q$: frozen copy, temperature 0.7); clean positions, OT pairing, weights as the plain model"
        return BECK
    if "denoiser" in t:
        f = DENOISER
        if "half model-chosen" in t:
            return f + r"; $x^l = x_1^l$ on $C$; half of the samples: $C = \{l : \max_i \hat D(I_t)^{l,i} > h\}$, $h$ = \cfg{loss.adaptive_h}; other half: $P(l \in C) = f \sim \mathcal{U}[0,1]$; loss over $l \notin C$"
        if "model-chosen" in t:
            return f + "; " + MODEL_C.replace(r"$p = \mathrm{sg}[\hat\delta_1(I_t)]$ ($\hat D(I_t)$ for a denoiser)", r"$p = \mathrm{sg}[\hat D(I_t)]$")
        if "random clean" in t or "partial context" in t:
            return f + "; " + RANDOM_C
        return f + r"; \cfg{train.steps} steps"
    if "fmlm" in t:
        return TGT["semigroup"].replace(r"\hat\delta_{\sigma_1}(x)", r"\hat\delta_{s,u}(x)").replace(r"\hat\delta_{\sigma_2}", r"\hat\delta_{u,t}").replace(r"\hat\Phi_{\sigma_1}", r"\hat\Phi_{s,u}") + r" (two-time network); " + PAIRS + r"; $\hat D$ = FLM"
    loss = "lagrangian" if "lagrangian" in t else "eulerian" if "eulerian" in t else "semigroup"
    parts = [TGT[loss]]
    dur = DUR["semigroup" if loss == "semigroup" else "other"]
    if "repository defaults" in t:
        parts.append(r"$t \sim \mathcal{U}[0,1]$, diagonal weight $\frac{1}{1-t}$; " + dur)
    elif "boundary term" in t or ("no auxiliary loss" in t and "durations" not in t) or (("partial context" in t) and "random clean" not in t):
        parts.append(PAIRS)
    else:
        parts.append(r"$\tau(t) \sim \mathcal{U}[0,1]$; " + dur + (r"; with probability $\frac{1}{32}$: $t = 0$, $\sigma = 1$" if "whole jump" in t or "two stages" in t or "targets" in t and loss == "semigroup" else ""))
    parts.append(r"$\bar D$ = frozen denoiser" if ("distilled" in t or "two stages" in t) else r"$\bar D = \hat\delta_0$")
    if "boundary term" in t:
        parts.append(r"auxiliary: on 10% of samples $t \sim \mathcal{U}[0.9,1]$, $\bar\delta = \bar D(x)$")
    if "model-chosen" in t:
        parts = [TGT[loss].split(", $\\gamma")[0], MODEL_C, parts[2]]
    elif "random clean" in t or "partial context" in t:
        short = r"clean positions $C$: $P(l \in C) = f \sim \mathcal{U}[0,1]$, losses over $l \notin C$"
        parts = [TGT[loss].split(", $\\gamma")[0], short, parts[2]] if "refinement" in t else [TGT[loss].split(", $\\gamma")[0], "time pairs as FMLM" if "partial context" in t else r"$\tau(t) \sim \mathcal{U}[0,1]$, " + dur, RANDOM_C, parts[2]]
    if "refinement" in t:
        parts.append(RIL + ("; " + SD if "self-distillation" in t else ""))
    if "without dropout" in t:
        parts.append("targets computed without dropout" + (", with the averaged weights" if "averaged" in t else ""))
    if "two stages" in t:
        parts.append("denoiser 3000 steps, then flow map 3000 steps")
    return "; ".join(parts)


def main():
    dry = "--dry" in sys.argv
    out = subprocess.run([NB, "runs"], capture_output=True, text=True).stdout
    n_ok = 0
    for line in out.splitlines():
        m = re.match(r"\s*#(\d+) (?:\(hidden\) )?\[(wt2-[a-z-]+)\] (.*?) · ", line)
        if not m:
            continue
        num, sub, title = int(m.group(1)), m.group(2), m.group(3)
        f = formula(title.split(": collapsed")[0])
        if "denoiser" not in title.lower():
            f = f.replace(r" ($\hat D(I_t)$ for a denoiser)", "")
        for cut in (r", $\gamma = \frac{\sigma_1(1-\sigma_2)}{\sigma_3}$", r"; $\sigma = \frac{t_e - t_s}{1 - t_s}$", r", losses over $l \notin C$", r"from step \cfg{loss.ril_start}, ", r" $p_k = \hat\delta_1(x_{(k-1)})$", " of samples"):
            if len(f) > 400:      # the dropped pieces are in the pseudocode of the algorithm
                f = f.replace(cut, "")
        if len(f) > 400:
            print(f"#{num} TOO LONG {len(f)}: {title}")
            continue
        if dry:
            print(f"#{num} [{len(f)}] {title}\n      {f}")
            continue
        r = subprocess.run([NB, "retitle", str(num), title, "--formula", f], capture_output=True, text=True)
        n_ok += "ok" in (r.stdout + r.stderr)
        if "ok" not in (r.stdout + r.stderr):
            print(f"#{num}: {(r.stdout + r.stderr).strip()[-200:]}")
    print("updated", n_ok)


main()
