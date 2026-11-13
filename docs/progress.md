# Progress log — Track A: NMF note transcription

*EEP55C36 final project · Smaragdis & Brown (2003) · individual project*

*Deadline: 2026-11-24*

This file records decisions, problems and next steps as they happen.

---

## 1. Project snapshot

| Item | Value |
|---|---|
| Track | A — Audio, generalized-KL NMF of magnitude spectrograms |
| Baseline | Multiplicative KL updates, frozen-dictionary inference |
| Modification | Temporal smoothness penalty on activations, weight $\mu$ |
| Primary metric (declared in advance) | Frame-level note $F_1$ on test clips, larger is better |
| Secondary metrics | $D/(FT)$ (unpenalized KL), onset smearing, runtime |
| Python / OS tested | _TODO_ |
| Repository URL | _TODO_ |
| Baseline tag / hash | _TODO_ |
| Submission tag / full hash | `v1.0-submission` / _TODO_ |

## 2. Scope statement

- **Reproduced:** generalized-KL NMF, $V \approx WH$, multiplicative updates, templates in columns of $W$, activations in rows of $H$.
- **Added protocol (not in the paper):** dictionary learned on development clips, frozen, activations inferred on held-out clips.
- **Also demonstrated:** blind joint factorization on a single clip (coinciding-notes merge).
- **Not reproduced:** the paper's real piano recording, Euclidean-cost experiments, energy-penalty variant (Eq. 4).
- **Paper quirks noted:** printed gradient labels in Eqs. (5)–(6) are dimensionally swapped; use explicit KL updates (Eq. 7 form).

## 3. Milestones

| Dates | Milestone | Status |
|---|---|---|
| 13–14 Nov | Repo created, plan committed, `data.py`, `baseline.py`, baseline tests, **baseline tag** | ☐ |
| 15–16 Nov | `alignment.py`, `metrics.py`, `protocol.py`, smoke config runs end to end | ☐ |
| 17–18 Nov | `modified.py`, modification tests, validation selection of $\mu$ and thresholds | ☐ |
| 19–20 Nov | Final experiments, ablation, stress cases, timing, figures | ☐ |
| 21–22 Nov | Report (2–3 pages), README, clean-clone rehearsal | ☐ |
| 23 Nov | Interview rehearsal, tag `v1.0-submission`, record hash | ☐ |
| 24 Nov | Submit (buffer day only) | ☐ |

## 4. Fixed protocol choices

Fill in and freeze **before** looking at test results.

| Choice | Value | Decided on |
|---|---|---|
| Number of components $K$ | 3 | |
| STFT | Hann, `nperseg=512`, `noverlap=384`, `fs=8000` | |
| Magnitude normalization | _one constant fixed from development data, or $\mu=\tilde\mu/\text{mean}(V)$_ | |
| Initialization | Positive, seeded, identical for both methods per clip | |
| Iterations (both methods) | _TODO_ (fixed count) | |
| `eps` floor | $10^{-10}$ | |
| Development seeds | _e.g. 0–9_ | |
| Validation seeds | _e.g. 100–109_ | |
| Test seeds | _e.g. 1000–1029_ (≥ 20 paired cases per setting) | |
| Difficulty settings | _three noise levels: low / mid / high_ | |
| Threshold grid (both methods) | _TODO_ | |
| $\mu$ grid (modified only) | _TODO_ | |
| Baseline extra search budget (to match $\mu$ tuning) | _TODO_ | |
| Component-to-pitch matching | Cosine similarity + one-to-one assignment, development data only | |
| Scale fixing | _per-component constant from development data_ | |
| Uncertainty | Mean paired difference with 95% CI, resampling over clips | |

## 5. Objective and modification (to fill in as derived)

Baseline objective:

$$D(V\,\|\,WH)=\sum_{f,t}\left(V_{ft}\log\frac{V_{ft}}{(WH)_{ft}}-V_{ft}+(WH)_{ft}\right)$$

Modified objective:

$$J(W,H)=D(V\,\|\,WH)+\frac{\mu}{2}\sum_{k}\sum_{t=2}^{T}\left(H_{kt}-H_{k,t-1}\right)^2$$

- **Update rule derived:** _TODO — positive/negative gradient split, boundary frames_
- **Monotonicity check:** _TODO — does $J$ decrease every iteration? If not, note and decide on the quadratic-root form_
- **Prior work cited:** _TODO — Virtanen (2007); verify citation details_
- **Hypothesis (written before experiments):**
  - Helps when: _TODO_
  - Hurts when: _TODO_
- **Extra cost introduced:** _TODO (expected small, $O(KT)$ per iteration)_

## 6. Verification checklist

- [ ] Hand-computable tiny example (one update step checked on paper)
- [ ] Limiting cases: $K=1$, exact rank-$K$ input, zero column in $V$, single-frame input
- [ ] Factors non-negative and finite; shapes correct
- [ ] Baseline KL non-increasing
- [ ] Inputs not mutated
- [ ] Template normalization leaves $WH$ unchanged
- [ ] Constant-in-time $H$ gives zero penalty
- [ ] Modified with $\mu=0$ equals baseline
- [ ] Optional cross-check against an external KL-NMF (validation only, cited, kept separate)

## 7. Experiment checklist

- [ ] Smoke config runs in about a minute and shows both methods
- [ ] Final config reproduces all tables and figures
- [ ] Three difficulty settings, ≥ 20 paired clips each
- [ ] Ablation: $\mu=0$ with threshold re-selected
- [ ] Extra control: baseline plus post-hoc smoothing filter
- [ ] Stress: shorter notes
- [ ] Stress: changed harmonic profile
- [ ] Stress: notes that always coincide in learning data
- [ ] Blind joint-factorization demo on one clip
- [ ] Timing: warm-up, repeats, setup reported separately, hardware recorded
- [ ] Figures: representative example, overall paired comparison, stress case

## 8. Decision log

*Newest first. One entry per decision.*

### YYYY-MM-DD — _title_
- **Context:** 
- **Options considered:** 
- **Decision:** 
- **Why:** 
- **Consequences / follow-ups:** 

## 9. Problems and fixes

| Date | Problem | Cause | Fix | Commit |
|---|---|---|---|---|
| | | | | |

## 10. Session notes

*Short dated entries: what was done, what is next.*

### 2026-11-13
- Read paper summary, brief and Track A guide.
- Drafted project structure and plan.
- **Next:** create repository, commit plan, start `data.py`.

## 11. Open questions

- [ ] _e.g. Is the simple split-gradient $H$ update monotone, or is the exact quadratic form needed?_
- [ ] _e.g. Which noise levels give a useful spread of baseline $F_1$?_
- [ ] 

## 12. Results summary

*Fill after the final run. State the primary-metric result honestly, including a null or negative one.*

- **Primary metric (mean paired difference, 95% CI) per setting:** _TODO_
- **Reconstruction loss $D/(FT)$:** _TODO_
- **Onset smearing:** _TODO_
- **Runtime overhead:** _TODO_
- **Ablation:** _TODO_
- **Stress cases:** _TODO_
- **Least convincing result / known limitations:** _TODO_

## 13. Pre-submission checklist

- [ ] README works from a clean clone: install, smoke run, tests, final run
- [ ] Report claims match saved results
- [ ] No test labels used in alignment, scaling or tuning
- [ ] Seeds, versions, hardware recorded in `results/environment.json`
- [ ] No paper PDFs, `.venv`, or secrets committed
- [ ] Teaching team has repository access
- [ ] Tag `v1.0-submission` created; full hash recorded in README, report and Section 1
- [ ] Assistance and attribution declaration written
- [ ] Interview backup (saved figures or screenshots) ready

## 14. Attribution and assistance

- **Paper:** P. Smaragdis and J. C. Brown (2003), WASPAA, 177–180.
- **External methods / code cited:** _TODO_
- **Collaboration:** _TODO_
- **AI assistance:** _TODO_
  - `progress.md` progress skeleton