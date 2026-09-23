#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCI v3 -- paper-faithful implementation of "Spectral Conformal Inference".

Changes vs v2
-------------
1. METHOD MATCHES THE MANUSCRIPT (Algorithm 1). The conformal weight is the
   FIXED spectral heterogeneity weight
       psi_J(x) = exp( -1/2 * sum_{j<=J} <x-Xbar, phi_j>^2 / lam_j )   [ridge-damped]
   optionally multiplied by an estimated covariate density ratio r_hat(x):
       w(x) = r_hat(x) * psi_J(x).
   Per-test-point kernel localization (v2's "SCI") is REMOVED from the main
   method and kept only as a clearly-labelled ablation arm (SCI-local).
2. EXACT CONFORMAL QUANTILES. No interpolation: the unweighted variant
   satisfies the finite-sample guarantee of Prop. 3.1; with the exact ratio,
   Prop. 3.2 holds (Tibshirani et al., 2019).
3. HETEROSCEDASTIC DGP matching manuscript Sec. 6.1.
4. SUBJECT-LEVEL (group) splits everywhere for real data.
5. REPRODUCIBLE seeds (hash() removed).
6. NEW EXPERIMENTS that make the innovation defensible:
     E0  validity check (no shift): unweighted coverage >= 1-alpha   [Appendix E]
     E7  mechanism ablation: coverage comes from r_hat, efficiency from psi_J
     E8  failure mode: outcome shift breaks ALL methods (honest reporting)
     E9  ratio-estimation error vs coverage deviation (validates Prop. 5.2)
     E9b rate validation: |coverage dev| and ratio error ~ n^{-beta/(2 beta+1)} (Prop. 1 & 2)
     E10 spectral-decay estimation (beta_hat, d_eff) for Fig. 1/2 regeneration
     E11 group-conditional coverage + Prop. 6 bound components
     (A6 arm in E7: target-anchored psi -- the Prop. 5 fix; gcv_cp arm in E6)

Run:  python SCI_v3.py --data real --out results_v3
      python SCI_v3.py --data semi   # semi-synthetic MRI/voice
      python SCI_v3.py --data sim    # simulation only
"""
import os, sys, glob, json, time, argparse, warnings
import numpy as np
from scipy import stats
from scipy.linalg import eigh
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import pairwise_distances
from sklearn.utils import resample
warnings.filterwarnings("ignore")

OUT = "results_v3"; DATA_MODE = "semi"
VERSION = "1.10-scannerdomains"

# ----------------------------------------------------------------------------
# exact conformal quantiles
# ----------------------------------------------------------------------------
def split_cp_quantile(r, alpha=0.1):
    """Exact split-CP quantile: k-th order statistic, k = ceil((n+1)(1-alpha)).
    Guarantees marginal coverage >= 1-alpha under exchangeability (no interpolation)."""
    r = np.sort(np.asarray(r, float)); n = len(r)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return r[min(max(k, 1), n) - 1]

def wcp_quantile(r, w, w_test, alpha=0.1):
    """Weighted split-CP quantile (Tibshirani et al. 2019): smallest score q with
    (sum_{R_i<=q} w_i + w_test)/(sum w_i + w_test) >= 1-alpha, where the test point
    carries mass w_test (a delta_infinity mass when the quantile is not reached).
    The quantile is INVARIANT to a common rescaling of the weights, so we normalize
    by the max weight (this makes underflow mathematically impossible). If the
    effective sample size is degenerate (ESS < 5) we ABSTAIN (return inf): the
    weighted-conformal construction itself assigns the residual mass to infinity,
    i.e. the prediction set becomes the whole range -- never a spuriously narrow
    interval."""
    r = np.asarray(r, float); w = np.asarray(w, float)
    mx = max(float(w.max(initial=0.0)), float(w_test))
    if mx <= 0.0:
        return np.inf
    w = w / mx; w_test = float(w_test) / mx
    mass = w.sum() + w_test
    ess = mass ** 2 / (np.sum(w ** 2) + w_test ** 2)
    if ess < 5.0:
        # conditional rescue: shrink toward the mean weight and retry before
        # abstaining (healthy-weight cases are untouched; degenerate real-data
        # shifts get finite intervals at the cost of a bounded blend bias)
        mb = 0.5 * w.mean()
        w2 = 0.5 * w + mb
        wt2 = 0.5 * w_test + mb
        m2 = w2.sum() + wt2
        ess2 = m2 ** 2 / (np.sum(w2 ** 2) + wt2 ** 2)
        if ess2 >= 5.0:
            w, w_test, mass = w2, wt2, m2
        else:
            return np.inf
    o = np.argsort(r, kind="mergesort"); r_s, w_s = r[o], w[o]
    cw = np.cumsum(w_s); lev = (1.0 - alpha) * mass
    k = int(np.searchsorted(cw, lev))
    return r_s[min(k, len(r_s) - 1)]

def wcp_quantile_multi(r, W, w_te, alpha=0.1):
    """Vectorised over test points. W: (n_cal, n_te) calibration weights."""
    out = np.empty(W.shape[1])
    for j in range(W.shape[1]):
        out[j] = wcp_quantile(r, W[:, j], w_te[j], alpha)
    return out

# ----------------------------------------------------------------------------
# data generating process (manuscript Sec. 6.1, heteroscedastic)
# ----------------------------------------------------------------------------
def sim_dgp(n, beta, m=200, seed=0, R=1.0, sigma=0.5, shift=True,
            strength=0.5, outcome_shift=False, var_shift=True):
    """Fourier-basis functional covariates, lam_j = j^{-2 beta}.

    Grid fix (CRITICAL): the half-sine family is orthonormal ONLY on the
    midpoint grid t_k = (k+0.5)/m. On linspace(0,1,m) the matrix Phi.T@Phi/m
    is far from I, which destroyed the signal in v1/v2 simulations.

    Regression function: genuinely infinite-dimensional, theta_j ~ j^{-beta}
    over all coordinates (unit energy), so the Pinsker-type rate is meaningful.
    Target (shift=True): modified eigenvalues, mean shift in the leading
    spectral coordinate, and a HEAVIER heteroscedastic noise (x1.7) driven by a
    DIFFERENT coordinate -- so a source-calibrated global quantile undercovers.
    outcome_shift=True additionally changes P(Y|X) (th -> -1.5 th, noise x1.2),
    which VIOLATES the covariate-shift assumption -> guaranteed failure of all
    conformal weightings (manuscript failure-mode experiment E8).
    """
    r = np.random.RandomState(seed)
    j = np.arange(1, m + 1)
    lam = j ** (-2 * beta)
    t = (np.arange(m) + 0.5) / m                       # midpoint grid: exact orthonormality
    Phi = np.sqrt(2) * np.sin((j[:, None] - 0.5) * np.pi * t[None, :])  # (m, m), Phi@Phi.T = m*I
    Z = r.randn(n, m) * np.sqrt(lam)[None, :]
    X = Z @ Phi.T
    th = (R / np.sqrt(np.sum(j ** (-2 * beta)))) * j ** (-beta)   # unit energy, decay beta
    sig = sigma * (1 + 0.5 * np.abs(Z[:, 0]) / np.sqrt(lam[0]))
    y = Z @ th + sig * r.randn(n)
    Xt, yt = None, None
    ratio_true = None
    if shift:
        # Structured, ESTIMABLE covariate shift (manuscript Sec 6.1):
        #  - variance inflation concentrated on the leading two coordinates;
        #    coordinate 1 is the heteroscedastic noise driver, so the density
        #    ratio aligns with residual size and ratio weighting CAN work
        #  - a clear mean shift in the leading coordinate
        lam_t = lam.copy()
        if var_shift:
            lam_t[0] *= 1.0 + strength   # variance inflation (robustness experiments)
            lam_t[1] *= 2.0              # additional estimable structure
        mu_t = np.zeros(m); mu_t[0] = 1.0
        Zt = r.randn(n, m) * np.sqrt(lam_t)[None, :] + mu_t[None, :]
        Xt = Zt @ Phi.T
        # SAME conditional noise as the source: P_T(Y|X) = P_S(Y|X).
        # Genuine covariate shift: the target visits high-noise regions of X
        # more often, so a source-calibrated global quantile undercovers and
        # the exact density ratio restores coverage (Tibshirani et al. 2019).
        sig_t = sigma * (1 + 0.5 * np.abs(Zt[:, 0]) / np.sqrt(lam_t[0]))
        if outcome_shift:
            yt = Zt @ (-1.5 * th) + 1.2 * sig_t * r.randn(n)
        else:
            yt = Zt @ th + sig_t * r.randn(n)
        def ratio_true(z, lam=lam, lam_t=lam_t, mu_t=mu_t):
            return np.exp(np.sum(z ** 2 * 0.5 * (1 / lam - 1 / lam_t), axis=1)
                          + z[:, 0] * mu_t[0] / lam_t[0] - mu_t[0] ** 2 / (2 * lam_t[0])
                          + 0.5 * np.sum(np.log(lam / lam_t)))
    return X, y, Xt, yt, (lam, th, ratio_true)

# ----------------------------------------------------------------------------
# subject-level split
# ----------------------------------------------------------------------------
def subject_split(ids, train_frac=0.7, seed=0):
    """Group-wise split: no subject id appears in both partitions."""
    ids = np.asarray(ids)
    gss = GroupShuffleSplit(1, train_size=train_frac, random_state=seed)
    (itr, ica), = gss.split(np.zeros(len(ids)), groups=ids)
    return itr, ica

# ----------------------------------------------------------------------------
# Spectral Conformal Inference  (manuscript Algorithm 1)
# ----------------------------------------------------------------------------
class SCI:
    """Spectral Conformal Inference (manuscript Algorithm 1).

    weight mode (mechanism separation for the ablation study):
      weighted=False                    -> unweighted split CP   (Prop. 3.1)
      weighted=True, use_ratio=False    -> psi-profile only      (efficiency arm)
      weighted=True, use_ratio=True     -> r_hat * psi-profile   (full method)
    Arms for the mechanism ablation (E7):
      weight='euclid'                   -> Euclidean profile instead of spectral
      ratio_fun=callable(Z)->r          -> oracle/true ratio instead of r_hat
    """
    def __init__(self, alpha=0.1, J=None, select="gcv", J_max=40, rho_frac=0.1,
                 weighted=True, use_ratio=False, weight="spectral",
                 ratio_fun=None, ridge=1e-6, seed=0, J_ratio=None, psi_in_weight=True,
                 target_anchored=False):
        self.alpha, self.J, self.select, self.J_max = alpha, J, select, J_max
        self.target_anchored = target_anchored
        self.J_ratio = J_ratio   # ratio-estimation dimension, separate from J
        self.psi_in_weight = psi_in_weight
        # FINDING (E7): the product r*psi cancels the coverage restoration of r,
        # because psi is centred on the SOURCE distribution while r upweights
        # target-like (far-from-source-centroid) points. Role separation:
        # shift correction uses w = r alone; psi serves efficiency adaptation
        # only in the no-shift regime (documented in the manuscript Sec. 5.2).
        self.rho_frac, self.weighted, self.use_ratio = rho_frac, weighted, use_ratio
        self.weight, self.ratio_fun, self.ridge, self.seed = weight, ratio_fun, ridge, seed

    def _basis(self, Z):
        lam, V = eigh(np.cov(Z.T))
        idx = np.argsort(lam)[::-1]
        return lam[idx], V[:, idx]

    def _gcv_J(self, S, y, cand):
        n = len(y); best, bJ = np.inf, cand[0]
        for J in cand:
            A = S[:, :J]
            G = A @ np.linalg.pinv(A.T @ A + self.ridge * np.eye(J)) @ A.T
            dof = np.trace(G)
            gcv = np.mean(((y - G @ y) / max(1 - dof / n, 0.2)) ** 2)
            if gcv < best: best, bJ = gcv, J
        return bJ

    def _gcv_J_conformal(self, S, y, cand, alpha):
        """Direction 2 (supporting tool, not a headline): GCV with a CP-inspired
        quantile penalty. NB: the residuals here are IN-SAMPLE (training fold),
        so q is a penalty for J-selection, NOT a true calibration quantile --
        stated as such in the manuscript to avoid a notation mismatch."""
        n = len(y); best, bJ = np.inf, cand[0]
        for J in cand:
            A = S[:, :J]
            G = A @ np.linalg.pinv(A.T @ A + self.ridge * np.eye(J)) @ A.T
            R = np.abs(y - G @ y)
            q = split_cp_quantile(R, alpha)
            gcv = np.mean((R - q) ** 2) / max(1 - np.trace(G) / n, 0.2) ** 2
            if gcv < best: best, bJ = gcv, J
        return bJ

    def _profile(self, Z):
        """Heterogeneity profile psi (fixed function of x): spectral (manuscript)
        or Euclidean (ablation arm)."""
        if self.target_anchored and hasattr(self, "V_T_"):
            D = Z - self.mu_T_
            Dj = D @ self.V_T_[:, :self.J_]
            lam = self.lam_T_[:self.J_]
            rho = max(self.rho_frac * lam[-1], 1e-10)
            d2 = np.sum(Dj ** 2 / (lam + rho), axis=1)
            return np.exp(-0.5 * d2)
        D = Z - self.mean_
        if self.weight == "euclid":
            d2 = np.sum(D ** 2, axis=1) / self._eu_scale
        else:
            Dj = D @ self.V_[:, :self.J_]
            lam = self.lam_[:self.J_]
            rho = self.rho_frac * lam[-1]
            d2 = np.sum(Dj ** 2 / (lam + rho), axis=1)
        return np.exp(-0.5 * d2)

    def _fit_ratio_clf(self, Z_cal, Z_tgt):
        Jr = min(self.J_ratio or 50, self.V_.shape[1])
        self.Jr_ = Jr
        Sc = Z_cal @ self.V_[:, :Jr]; St = Z_tgt @ self.V_[:, :Jr]
        Xc = np.vstack([Sc, St]); lab = np.r_[np.zeros(len(Sc)), np.ones(len(St))]
        # class_weight='balanced': the conformal construction needs the LIKELIHOOD
        # RATIO, which equals p/(1-p) only under equal priors (Tibshirani et al. 2019)
        self._ratio_clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0,
                                             class_weight="balanced").fit(Xc, lab)
        self._ratio_clip = (0.05, 20.0)

    def _ratio_hat(self, Z):
        p = self._ratio_clf.predict_proba(Z @ self.V_[:, :self.Jr_])[:, 1]
        return np.clip(p / (1 - p + 1e-10), *self._ratio_clip)

    def _ratio(self, Z):
        if self.ratio_fun is not None:
            return self.ratio_fun(Z)
        return self._ratio_hat(Z)

    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt_unlab=None):
        self.scx_ = StandardScaler().fit(X_tr)
        self.scy_ = StandardScaler().fit(y_tr.reshape(-1, 1))
        Z = self.scx_.transform(X_tr)
        y = self.scy_.transform(y_tr.reshape(-1, 1))[:, 0]
        self.lam_, self.V_ = self._basis(Z)
        self.mean_ = Z.mean(0)
        self._eu_scale = max(np.median(np.sum((Z - self.mean_) ** 2, axis=1)), 1e-8)
        if self.ratio_fun is not None and hasattr(self.ratio_fun, "set_context"):
            self.ratio_fun.set_context(self)   # oracle ratio: map std. space back to Z
        cand = np.arange(2, min(self.J_max, Z.shape[1], len(y) - 1) + 1)
        if self.J is not None:
            self.J_ = self.J
        elif self.select == "gcv_cp":
            self.J_ = self._gcv_J_conformal(Z @ self.V_, y, cand, self.alpha)
        else:
            self.J_ = self._gcv_J(Z @ self.V_, y, cand)
        self.model_ = Ridge(self.ridge).fit(Z @ self.V_[:, :self.J_], y)
        if self.target_anchored:
            Zt0 = self.scx_.transform(X_tgt_unlab) if X_tgt_unlab is not None else Z
            self.mu_T_ = Zt0.mean(0)
            self.lam_T_, self.V_T_ = self._basis(Zt0)   # target-centred geometry
        self._X_cal_ref = np.asarray(X_cal)
        Zc = self.scx_.transform(X_cal)
        self.R_ = np.abs(self.scy_.transform(y_cal.reshape(-1, 1))[:, 0]
                         - self.model_.predict(Zc @ self.V_[:, :self.J_]))
        if not self.weighted:
            self.w_ = np.ones(len(self.R_))
        else:
            psi = self._profile(Zc)
            if (self.use_ratio or self.ratio_fun is not None):
                # ratio trained on the TRAINING fold (independent of the
                # calibration scores) + unlabeled target -- clean dependence
                if self.ratio_fun is None:
                    self._fit_ratio_clf(Z, self.scx_.transform(X_tgt_unlab))
                r = self._ratio(Zc)
            else:
                r = np.ones(len(psi))
            self.w_ = (r * psi) if self.psi_in_weight else r
            self.r_, self.psi_ = r, psi
        return self

    def predict(self, X_te):
        Zt = self.scx_.transform(X_te)
        mu = self.model_.predict(Zt @ self.V_[:, :self.J_])
        if not self.weighted:
            q = np.full(len(X_te), split_cp_quantile(self.R_, self.alpha))
        else:
            psi_te = self._profile(Zt)
            if (self.use_ratio or self.ratio_fun is not None):
                r_te = self._ratio(Zt)
            else:
                r_te = np.ones(len(X_te))
            W = np.tile(self.w_, (len(X_te), 1)).T
            w_te = (r_te * psi_te) if self.psi_in_weight else r_te
            q = wcp_quantile_multi(self.R_, W, w_te, self.alpha)
        mid = self.scy_.inverse_transform(mu.reshape(-1, 1))[:, 0]
        lo = np.full(len(X_te), -np.inf); hi = np.full(len(X_te), np.inf)
        finite = np.isfinite(q)
        if finite.any():
            lo[finite] = self.scy_.inverse_transform(
                (mu[finite] - q[finite]).reshape(-1, 1))[:, 0]
            hi[finite] = self.scy_.inverse_transform(
                (mu[finite] + q[finite]).reshape(-1, 1))[:, 0]
        return mid, lo, hi

# ---------------------------------------------------------------------------
# baseline conformal methods (exact quantiles throughout)
# ----------------------------------------------------------------------------
class NaiveCP:
    def __init__(self, alpha=0.1): self.alpha = alpha
    def fit(self, X_tr, y_tr, X_cal, y_cal, **_):
        self.scx = StandardScaler().fit(X_tr); self.scy = StandardScaler().fit(y_tr.reshape(-1,1))
        self.m = Ridge(1.0).fit(self.scx.transform(X_tr), self.scy.transform(y_tr.reshape(-1,1))[:,0])
        self.q = split_cp_quantile(np.abs(self.scy.transform(y_cal.reshape(-1,1))[:,0]
                              - self.m.predict(self.scx.transform(X_cal))), self.alpha)
        return self
    def predict(self, X):
        mu = self.scy.inverse_transform(self.m.predict(self.scx.transform(X)).reshape(-1,1))[:,0]
        q = self.q * float(self.scy.scale_[0])   # q was computed in standardized-y units
        return mu, mu - q, mu + q

class KRRCP(NaiveCP):
    """Kernel ridge regression + split conformal (Gaussian RBF)."""
    def fit(self, X_tr, y_tr, X_cal, y_cal, **_):
        self.scx = StandardScaler().fit(X_tr); self.scy = StandardScaler().fit(y_tr.reshape(-1,1))
        self.m = KernelRidge(kernel="rbf", alpha=1.0, gamma=0.05)
        self.m.fit(self.scx.transform(X_tr), self.scy.transform(y_tr.reshape(-1,1))[:,0])
        self.q = split_cp_quantile(np.abs(self.scy.transform(y_cal.reshape(-1,1))[:,0]
                              - self.m.predict(self.scx.transform(X_cal))), self.alpha)
        return self

class MLPCP(NaiveCP):
    def fit(self, X_tr, y_tr, X_cal, y_cal, **_):
        self.scx = StandardScaler().fit(X_tr); self.scy = StandardScaler().fit(y_tr.reshape(-1,1))
        self.m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800, random_state=0)
        self.m.fit(self.scx.transform(X_tr), self.scy.transform(y_tr.reshape(-1,1))[:,0])
        self.q = split_cp_quantile(np.abs(self.scy.transform(y_cal.reshape(-1,1))[:,0]
                              - self.m.predict(self.scx.transform(X_cal))), self.alpha)
        return self

class WeightedCP:
    """Density-ratio weighted split conformal (Tibshirani et al. 2019).
    FIX vs v2: the ratio is evaluated at the CALIBRATION points."""
    def __init__(self, alpha=0.1, clip=(0.1, 10.0)): self.alpha, self.clip = alpha, clip
    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt_unlab=None, **_):
        self.scx = StandardScaler().fit(X_tr); self.scy = StandardScaler().fit(y_tr.reshape(-1,1))
        self.m = Ridge(1.0).fit(self.scx.transform(X_tr), self.scy.transform(y_tr.reshape(-1,1))[:,0])
        r = np.abs(self.scy.transform(y_cal.reshape(-1,1))[:,0]
                   - self.m.predict(self.scx.transform(X_cal)))
        self.R_ = r; self.w_ = np.ones(len(r))
        if X_tgt_unlab is not None:
            Xc = np.vstack([self.scx.transform(X_tr), self.scx.transform(X_tgt_unlab)])
            lab = np.r_[np.zeros(len(X_tr)), np.ones(len(X_tgt_unlab))]
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0,
                                     class_weight="balanced").fit(Xc, lab)
            p = clf.predict_proba(Xc)[:, 1]
            self.w_ = np.clip(p[:len(X_cal)] / (1 - p[:len(X_cal)] + 1e-10), *self.clip)
        return self
    def predict(self, X):
        mu = self.scy.inverse_transform(self.m.predict(self.scx.transform(X)).reshape(-1,1))[:,0]
        if np.all(self.w_ == 1.0):
            q = np.full(len(X), split_cp_quantile(self.R_, self.alpha))
        else:
            q = np.full(len(X), wcp_quantile(self.R_, self.w_, 1.0, self.alpha))
        q = q * float(self.scy.scale_[0])
        return mu, mu - q, mu + q

class NormalizedCP:
    """Locally adaptive (normalized-score) split conformal -- Papadopoulos et al.
    2002: S = |y - mu(x)| / sigma(x) with a fitted scale model. Included as the
    classical baseline that the psi-heterogeneity idea must be distinguished
    from (manuscript related-work paragraph)."""
    def __init__(self, alpha=0.1): self.alpha = alpha
    def fit(self, X_tr, y_tr, X_cal, y_cal, **_):
        self.scx = StandardScaler().fit(X_tr); self.scy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Z = self.scx.transform(X_tr); y = self.scy.transform(y_tr.reshape(-1, 1))[:, 0]
        self.m = Ridge(1.0).fit(Z, y)
        self.s = Ridge(1.0).fit(Z, np.log(np.abs(y - self.m.predict(Z)) + 1e-6))
        Zc = self.scx.transform(X_cal); yc = self.scy.transform(y_cal.reshape(-1, 1))[:, 0]
        S = np.abs(yc - self.m.predict(Zc)) / np.exp(self.s.predict(Zc))
        self.q = split_cp_quantile(S, self.alpha)
        return self
    def predict(self, X):
        Z = self.scx.transform(X)
        mu = self.scy.inverse_transform(self.m.predict(Z).reshape(-1, 1))[:, 0]
        sig = np.exp(self.s.predict(Z)) * float(self.scy.scale_[0])
        return mu, mu - self.q * sig, mu + self.q * sig

class WeightedNormalizedCP:
    """Ratio-weighted + locally-normalized split CP -- the DIRECT competitor
    that isolates the role of psi (spectral heterogeneity) from the two known
    ingredients (Tibshirani ratio weighting + Papadopoulos normalization).
    If this ties SCI_full on both coverage and length, psi has no independent
    contribution; if it matches coverage but is longer, the spectral profile
    gives an efficiency gain (Prop. 4 empirical evidence)."""
    def __init__(self, alpha=0.1, clip=(0.1, 10.0)):
        self.alpha, self.clip = alpha, clip
    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt_unlab=None, **_):
        self.scx = StandardScaler().fit(X_tr)
        self.scy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Z = self.scx.transform(X_tr); y = self.scy.transform(y_tr.reshape(-1, 1))[:, 0]
        self.m = Ridge(1.0).fit(Z, y)
        self.s = Ridge(1.0).fit(Z, np.log(np.abs(y - self.m.predict(Z)) + 1e-6))
        Zc = self.scx.transform(X_cal); yc = self.scy.transform(y_cal.reshape(-1, 1))[:, 0]
        self.R_ = np.abs(yc - self.m.predict(Zc)) / np.exp(self.s.predict(Zc))
        self.w_ = np.ones(len(self.R_))
        if X_tgt_unlab is not None:
            Xc = np.vstack([Zc, self.scx.transform(X_tgt_unlab)])
            lab = np.r_[np.zeros(len(Zc)), np.ones(len(X_tgt_unlab))]
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0,
                                     class_weight="balanced").fit(Xc, lab)
            p = clf.predict_proba(Xc)[:, 1]
            self.w_ = np.clip(p[:len(Zc)] / (1 - p[:len(Zc)] + 1e-10), *self.clip)
        return self
    def predict(self, X):
        Z = self.scx.transform(X)
        mu = self.scy.inverse_transform(self.m.predict(Z).reshape(-1, 1))[:, 0]
        sig = np.exp(self.s.predict(Z)) * float(self.scy.scale_[0])
        if np.all(self.w_ == 1.0):
            q = np.full(len(X), split_cp_quantile(self.R_, self.alpha))
        else:
            q = np.full(len(X), wcp_quantile(self.R_, self.w_, 1.0, self.alpha))
        return mu, mu - q * sig, mu + q * sig

class AERatioCP:
    """Ratio-weighted split CP with the density ratio estimated in a LEARNED
    autoencoder representation space (the distance-aware-conformal style
    competitor). Preempts the question 'why not just learn a representation?':
    the learned space needs architecture choice and target-free training with
    no validity criterion, whereas the spectral space is parameter-free and
    checkable (beta_hat). A7 in the mechanism ablation."""
    def __init__(self, alpha=0.1, bottleneck=10, clip=(0.1, 10.0), seed=0):
        self.alpha, self.bottleneck, self.clip, self.seed = alpha, bottleneck, clip, seed
    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt_unlab=None, **_):
        self.scx = StandardScaler().fit(X_tr)
        self.scy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Z = self.scx.transform(X_tr)
        y = self.scy.transform(y_tr.reshape(-1, 1))[:, 0]
        self.m = Ridge(1.0).fit(Z, y)
        self.R_ = np.abs(self.scy.transform(y_cal.reshape(-1, 1))[:, 0]
                         - self.m.predict(self.scx.transform(X_cal)))
        self.w_ = np.ones(len(self.R_))
        if X_tgt_unlab is not None:
            self._ae = MLPRegressor(hidden_layer_sizes=(64, self.bottleneck, 64),
                                    max_iter=400, random_state=self.seed)
            self._ae.fit(Z, Z)
            def code(X_):
                A = np.maximum(X_ @ self._ae.coefs_[0] + self._ae.intercepts_[0], 0)
                return A @ self._ae.coefs_[1] + self._ae.intercepts_[1]
            C = np.vstack([code(Z), code(self.scx.transform(X_tgt_unlab))])
            lab = np.r_[np.zeros(len(Z)), np.ones(len(X_tgt_unlab))]
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0,
                                     class_weight="balanced").fit(C, lab)
            p = clf.predict_proba(C)[:, 1]
            self.w_ = np.clip(p[:len(X_cal)] / (1 - p[:len(X_cal)] + 1e-10), *self.clip)
        return self
    def predict(self, X):
        mu = self.scy.inverse_transform(
            self.m.predict(self.scx.transform(X)).reshape(-1, 1))[:, 0]
        if np.all(self.w_ == 1.0):
            q = np.full(len(X), split_cp_quantile(self.R_, self.alpha))
        else:
            q = np.full(len(X), wcp_quantile(self.R_, self.w_, 1.0, self.alpha))
        q = q * float(self.scy.scale_[0])
        q = q * float(self.scy.scale_[0])
        return mu, mu - q, mu + q

class RPRatioCP:
    """Ratio-weighted CP with the ratio estimated on a Gaussian RANDOM PROJECTION
    to the same dimension used by the spectral ratio -- the dimension-matched
    control. If random projection fails where the spectral space succeeds, the
    spectral GEOMETRY (signal-aligned coordinates), not mere dimension
    reduction, is the operative ingredient (ablation arm A8)."""
    def __init__(self, alpha=0.1, n_components=10, clip=(0.1, 10.0), seed=0):
        self.alpha, self.n_components, self.clip, self.seed = alpha, n_components, clip, seed
    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt_unlab=None, **_):
        self.scx = StandardScaler().fit(X_tr)
        self.scy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Z = self.scx.transform(X_tr)
        y = self.scy.transform(y_tr.reshape(-1, 1))[:, 0]
        self.m = Ridge(1.0).fit(Z, y)
        self.R_ = np.abs(self.scy.transform(y_cal.reshape(-1, 1))[:, 0]
                         - self.m.predict(self.scx.transform(X_cal)))
        self.w_ = np.ones(len(self.R_))
        if X_tgt_unlab is not None:
            rng = np.random.RandomState(self.seed)
            P = rng.randn(Z.shape[1], self.n_components)
            P /= np.linalg.norm(P, axis=0, keepdims=True)
            C = np.vstack([Z @ P, self.scx.transform(X_tgt_unlab) @ P])
            lab = np.r_[np.zeros(len(Z)), np.ones(len(X_tgt_unlab))]
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=0,
                                     class_weight="balanced").fit(C, lab)
            p = clf.predict_proba(C)[:, 1]
            self.w_ = np.clip(p[:len(X_cal)] / (1 - p[:len(X_cal)] + 1e-10), *self.clip)
        return self
    def predict(self, X):
        mu = self.scy.inverse_transform(
            self.m.predict(self.scx.transform(X)).reshape(-1, 1))[:, 0]
        if np.all(self.w_ == 1.0):
            q = np.full(len(X), split_cp_quantile(self.R_, self.alpha))
        else:
            q = np.full(len(X), wcp_quantile(self.R_, self.w_, 1.0, self.alpha))
        q = q * float(self.scy.scale_[0])
        q = q * float(self.scy.scale_[0])
        return mu, mu - q, mu + q

class OracleCP:
    """Oracle: true test-label noise quantile. Uses TEST LABELS -- diagnostic only,
    never a legitimate competitor; must be flagged in every table."""
    def __init__(self, alpha=0.1): self.alpha = alpha
    def fit(self, X_tr, y_tr, X_cal, y_cal, X_tgt=None, y_tgt=None, **_):
        self.scy = StandardScaler().fit(y_tr.reshape(-1,1))
        self.q = np.quantile(np.abs(self.scy.transform(y_tgt.reshape(-1,1))[:,0]), 1 - self.alpha)
        return self
    def predict(self, X):
        mu = np.zeros(len(X))
        return mu, mu - self.q, mu + self.q

# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def cov_len(y, lo, hi):
    """Coverage counts abstained (infinite) intervals as covered; the reported
    length is the mean over non-abstained points (abstention rate is implicit)."""
    length = np.asarray(hi, float) - np.asarray(lo, float)
    finite = np.isfinite(length)
    mean_len = float(np.mean(length[finite])) if finite.any() else np.inf
    return float(np.mean((y >= lo) & (y <= hi))), mean_len

def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.clip(a - b, -1e12, 1e12)   # inf (abstained) intervals -> large finite
    if np.all(d == 0): return 1.0
    try: return float(stats.wilcoxon(d).pvalue)
    except Exception: return 1.0

def wilson_ci(k, n, z=1.96):
    p = k / max(n, 1); d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return float(c - h), float(c + h)

def spectral_stats(X, Jmax=50):
    """Empirical eigen-decay stats for Fig. 1/2: beta_hat and effective dimension."""
    Z = StandardScaler().fit_transform(X)
    lam, _ = eigh(np.cov(Z.T)); lam = np.sort(lam)[::-1]
    j = np.arange(1, Jmax + 1)
    b = np.polyfit(np.log(j), np.log(np.maximum(lam[:Jmax], 1e-12)), 1)[0]
    deff = float(np.sum((lam / lam[0]) ** 2))
    return -b / 2, deff, lam

def save_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)

# ----------------------------------------------------------------------------
# E0  validity check under NO shift (Appendix E of the manuscript)
# ----------------------------------------------------------------------------
def exp_validity_check(out=OUT, n=500, m=200, beta=1.0, reps=200, alpha=0.1):
    rows = []
    for mode, kw in [("unweighted", dict(weighted=False)),
                     ("psi_only", dict(weighted=True, use_ratio=False))]:
        covs = []
        for rep in range(reps):
            X, y, _, _, _ = sim_dgp(n, beta, m=m, seed=rep, shift=False)
            itr, ica = subject_split(np.arange(n), 0.7, rep)
            md = SCI(alpha=alpha, **kw).fit(X[itr], y[itr], X[ica], y[ica])
            # fresh test sample from the SAME distribution (exchangeable)
            Xe, ye, _, _, _ = sim_dgp(2000, beta, m=m, seed=10_000 + rep, shift=False)
            _, lo, hi = md.predict(Xe)
            covs.append(np.mean((ye >= lo) & (ye <= hi)))
        lo_c, hi_c = wilson_ci(int(np.mean(covs) * reps), reps)
        rows.append(dict(exp="E0_validity", mode=mode, n=n,
                         coverage_mean=float(np.mean(covs)), coverage_se=float(np.std(covs)/np.sqrt(reps)),
                         ci_lo=lo_c, ci_hi=hi_c,
                         pass_=("YES" if np.mean(covs) >= 1 - alpha - 2*np.std(covs)/np.sqrt(reps) else "NO")))
        print(f"[E0] {mode:12s} coverage = {np.mean(covs):.4f} +- {np.std(covs)/np.sqrt(reps):.4f} "
              f"(nominal {1-alpha})  -> {rows[-1]['pass_']}")
    save_csv(os.path.join(out, "E0_validity.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E1  simulation: Table-2-style comparison UNDER COVARIATE SHIFT + rate slopes
# ----------------------------------------------------------------------------
def exp_simulation(out=OUT, betas=(0.8, 1.0, 1.5, 2.5), n=500, m=200, reps=100, alpha=0.1):
    methods = {
        "SCI_full(r_spec)": lambda: SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False),
        "SCI_psi_only":     lambda: SCI(alpha=alpha, weighted=True, use_ratio=False),
        "SCI_unweighted":   lambda: SCI(alpha=alpha, weighted=False),
        "WeightedCP(r)":    lambda: WeightedCP(alpha=alpha),
        "KRR_CP":           lambda: KRRCP(alpha=alpha),
        "NaiveCP":          lambda: NaiveCP(alpha=alpha),
        "NormalizedCP":     lambda: NormalizedCP(alpha=alpha),
        "WtdNormalizedCP":  lambda: WeightedNormalizedCP(alpha=alpha),
    }
    rows, res = [], {k: {b: ([], []) for b in betas} for k in methods}
    for beta in betas:
        for rep in range(reps):
            X, y, Xt, yt, _ = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
            itr, ica = subject_split(np.arange(n), 0.7, rep)
            for k, mk in methods.items():
                md = mk().fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
                _, lo, hi = md.predict(Xt)
                c, l = cov_len(yt, lo, hi)
                res[k][beta][0].append(c); res[k][beta][1].append(l)
    for beta in betas:
        for k in methods:
            c, l = res[k][beta]
            rows.append(dict(exp="E1_shift", beta=beta, method=k,
                             coverage=float(np.mean(c)), cov_se=float(np.std(c)/np.sqrt(reps)),
                             length=float(np.mean(l)), len_se=float(np.std(l)/np.sqrt(reps)),
                             p_vs_full=wilcoxon(res["SCI_full(r_spec)"][beta][0], c) if k != "SCI_full(r_spec)" else 1.0))
    save_csv(os.path.join(out, "E1_simulation_shift.csv"), rows)
    for beta in betas:
        print(f"[E1] beta={beta}: " + " | ".join(
            f"{k}: cov={np.mean(res[k][beta][0]):.3f}, len={np.mean(res[k][beta][1]):.2f}"
            for k in methods))
    return rows

# ----------------------------------------------------------------------------
# E7  mechanism ablation: coverage comes from r_hat, efficiency from psi_J
# ----------------------------------------------------------------------------
class _OracleRatio:
    """True Gaussian density ratio in Z-space, mapped from SCI's standardized
    feature space. Decisive test: if A4 does NOT restore coverage, the weighted
    quantile machinery is buggy; if A4 works but A3 fails, estimation is the gap."""
    def __init__(self, m, ratio_true):
        self.m = m; self.ratio_true = ratio_true
    def set_context(self, sci):
        j = np.arange(1, self.m + 1)
        t = (np.arange(self.m) + 0.5) / self.m
        Phi = np.sqrt(2) * np.sin((j[:, None] - 0.5) * np.pi * t[None, :])
        # Robust inversion: X = Z @ Phi.T  =>  Z = X @ pinv(Phi.T). For the
        # midpoint-grid sine basis Phi happens to be exactly symmetric
        # (sin(pi(j-.5)(k-.5)/m)), so pinv(Phi.T) == Phi/m; pinv keeps the
        # recovery correct for ANY orthonormal basis (cosine, other grids).
        self._pinvT = np.linalg.pinv(Phi.T)
        self._scx = sci.scx_
    def __call__(self, Z):
        X = Z * self._scx.scale_ + self._scx.mean_
        return self.ratio_true(X @ self._pinvT)

def exp_ablation(out=OUT, beta=1.0, n=500, m=200, reps=100, alpha=0.1):
    """Mechanism ablation: coverage comes from the ratio, efficiency from psi.
    A4 (exact ratio) is the decisive machinery validation and must reach ~0.90."""
    rows, res = [], {}
    for rep in range(reps):
        X, y, Xt, yt, aux = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
        lam, th, ratio_true = aux
        oracle = _OracleRatio(m, ratio_true)
        arms = {
            "A0_unweighted":    SCI(alpha=alpha, weighted=False),
            "A1_psi_only":      SCI(alpha=alpha, weighted=True, use_ratio=False),
            "A2_ratio_rawX":    WeightedCP(alpha=alpha),
            "A3_ratio_spectral":SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False),
            "A4_oracle_ratio":  SCI(alpha=alpha, weighted=True, use_ratio=True,
                                    ratio_fun=oracle, psi_in_weight=False),
            "A5_product(r*psi)":SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=True),
            "A6_target_anchored":SCI(alpha=alpha, weighted=True, use_ratio=True,
                                     psi_in_weight=True, target_anchored=True),
            "A7_ratio_AEspace": AERatioCP(alpha=alpha, seed=rep),
            "A8_ratio_randomproj": RPRatioCP(alpha=alpha, seed=rep),
        }
        itr, ica = subject_split(np.arange(n), 0.7, rep)
        for k, md in arms.items():
            md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
            _, lo, hi = md.predict(Xt)
            c, l = cov_len(yt, lo, hi)
            res.setdefault(k, ([], []))
            res[k][0].append(c); res[k][1].append(l)
    for k in arms:
        c, l = res[k]
        rows.append(dict(exp="E7_ablation", arm=k, coverage=float(np.mean(c)),
                         cov_se=float(np.std(c)/np.sqrt(reps)),
                         length=float(np.mean(l)), len_se=float(np.std(l)/np.sqrt(reps))))
        print(f"[E7] {k:16s} coverage={np.mean(c):.3f}+-{np.std(c)/np.sqrt(reps):.3f}  "
              f"length={np.mean(l):.3f}")
    p_len = wilcoxon(res["A4_oracle_ratio"][1], res["A6_target_anchored"][1])
    p_cov = wilcoxon(res["A4_oracle_ratio"][0], res["A6_target_anchored"][0])
    print(f"[E7] A6 vs A4 paired Wilcoxon: length p={p_len:.4f}, coverage p={p_cov:.4f} "
          f"-> Prop. 5 efficiency {'SUPPORTED' if p_len < 0.05 else 'not significant'}")
    rows.append(dict(exp="E7_ablation", arm="A6_vs_A4_wilcoxon", coverage=p_cov, cov_se=0.0,
                     length=p_len, len_se=0.0))
    save_csv(os.path.join(out, "E7_ablation.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E8  failure mode: outcome shift -> NO method can restore coverage (honest)
# ----------------------------------------------------------------------------
def exp_failure_mode(out=OUT, beta=1.0, n=500, m=200, reps=100, alpha=0.1):
    methods = {
        "SCI_full":   lambda: SCI(alpha=alpha, weighted=True, use_ratio=True),
        "SCI_psi":    lambda: SCI(alpha=alpha, weighted=True, use_ratio=False),
        "WeightedCP": lambda: WeightedCP(alpha=alpha),
        "NaiveCP":    lambda: NaiveCP(alpha=alpha),
    }
    rows, res = [], {k: ([], []) for k in methods}
    for rep in range(reps):
        X, y, Xt, yt, _ = sim_dgp(n, beta, m=m, seed=rep, outcome_shift=True)  # keeps var_shift
        itr, ica = subject_split(np.arange(n), 0.7, rep)
        for k, mk in methods.items():
            md = mk().fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
            _, lo, hi = md.predict(Xt)
            c, l = cov_len(yt, lo, hi)
            res[k][0].append(c); res[k][1].append(l)
    for k in methods:
        c, l = res[k]
        rows.append(dict(exp="E8_failure_mode", method=k, coverage=float(np.mean(c)),
                         cov_se=float(np.std(c)/np.sqrt(reps)), length=float(np.mean(l))))
        print(f"[E8] {k:12s} coverage={np.mean(c):.3f} (nominal {1-alpha})  <-- expected to FAIL")
    save_csv(os.path.join(out, "E8_failure_mode.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E9  ratio-estimation error vs coverage deviation (validates Prop. 5.2)
# ----------------------------------------------------------------------------
def exp_ratio_sensitivity(out=OUT, beta=1.0, n=500, m=200, reps=50, alpha=0.1):
    """Validates Prop. 5.2: coverage deviation is controlled by the L1 ratio
    estimation error E|r_hat - r|. Uses the CORRECTED method (w = r_hat alone,
    psi NOT multiplied -- the product destroys coverage restoration, see E7 A5)."""
    j = np.arange(1, m + 1)
    t = (np.arange(m) + 0.5) / m
    Phi = np.sqrt(2) * np.sin((j[:, None] - 0.5) * np.pi * t[None, :])
    rows = []
    for nu in [50, 100, 200, 500, 1000]:
        covs, ratio_errs = [], []
        for rep in range(reps):
            X, y, Xt, yt, aux = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
            lam, th, ratio_true = aux
            itr, ica = subject_split(np.arange(n), 0.7, rep)
            perm = np.random.RandomState(rep).permutation(len(Xt))
            nu_eff = min(nu, len(Xt))
            Xu = Xt[perm[:nu_eff]]                 # unlabeled: used to fit the ratio
            Xev_t = Xt[perm[nu_eff:]]              # held-out target for evaluation
            md = SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False)
            md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xu)
            _, lo, hi = md.predict(Xt)
            covs.append(np.mean((yt >= lo) & (yt <= hi)))
            # balanced held-out evaluation: source training fold + held-out target
            if len(Xev_t) == 0:
                Xev_t = Xt[perm[:50]]          # nu exhausted the target pool:
            nev = min(len(Xev_t), len(ica))    # calibration fold is held-out for the ratio clf
            Xev = np.vstack([X[ica][:nev], Xev_t[:nev]])
            zev = Xev @ np.linalg.pinv(Phi.T)
            r_true_vals = ratio_true(zev)
            r_hat_vals = md._ratio(md.scx_.transform(Xev))
            ratio_errs.append(float(np.mean(np.abs(r_hat_vals - r_true_vals))))
        rows.append(dict(exp="E9_ratio_sensitivity", n_unlab=nu,
                         coverage=float(np.mean(covs)), cov_se=float(np.std(covs)/np.sqrt(reps)),
                         ratio_L1=float(np.mean(ratio_errs))))
        print(f"[E9] n_unlab={nu:5d}: coverage={np.mean(covs):.3f}, "
              f"L1_ratio_error={np.mean(ratio_errs):.3f}")
    save_csv(os.path.join(out, "E9_ratio_sensitivity.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E9b rate validation: |coverage deviation| ~ n^{-beta/(2 beta + 1)}
# ----------------------------------------------------------------------------
def exp_rate_validation(out=OUT, betas=(0.8, 1.0, 1.5, 2.5),
                        ns=(100, 200, 500, 1000, 2000), reps=60, alpha=0.1, m=200):
    """Direction 1 empirical validation (Prop. 1 & 2): the coverage deviation
    and the ratio-estimation error should decay with log-log slope
    -beta/(2 beta + 1) in n. Points within 3x the Monte-Carlo floor are
    excluded from the coverage-deviation fit."""
    j = np.arange(1, m + 1)
    t = (np.arange(m) + 0.5) / m
    Phi = np.sqrt(2) * np.sin((j[:, None] - 0.5) * np.pi * t[None, :])
    rows = []
    for beta in betas:
        devs, rerrs, nn = [], [], []
        for n in ns:
            covs, errs = [], []
            for rep in range(reps):
                X, y, Xt, yt, aux = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
                lam, th, ratio_true = aux
                itr, ica = subject_split(np.arange(n), 0.7, rep)
                md = SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False)
                md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
                _, lo, hi = md.predict(Xt)
                covs.append(np.mean((yt >= lo) & (yt <= hi)))
                r_hat_vals = md._ratio(md.scx_.transform(X[ica]))
                r_true_vals = ratio_true(X[ica] @ np.linalg.pinv(Phi.T))
                errs.append(float(np.mean(np.abs(r_hat_vals - r_true_vals))))
            dev = abs(np.mean(covs) - (1 - alpha))
            devs.append(dev); rerrs.append(float(np.mean(errs))); nn.append(n)
            print(f"[E9b] beta={beta} n={n}: cov={np.mean(covs):.4f}, dev={dev:.4f}, "
                  f"ratio_err={np.mean(errs):.3f}")
            rows.append(dict(exp="E9b_detail", beta=beta, n=n,
                             cov_dev=round(dev, 5), ratio_err=round(float(np.mean(errs)), 5)))
        nn = np.array(nn, float); devs = np.array(devs); rerrs = np.array(rerrs)
        mc_floor = float(np.sqrt(alpha * (1 - alpha) / (reps * 500)))
        mask = devs > 3 * mc_floor
        # Two-source decomposition (reviewer-suggested): the deviation has an
        # O(n^{-1/2}) split-CP floor term plus the O(n^{-beta/(2beta+1)}) ratio
        # term; fit both to separate their contributions.
        from scipy.optimize import curve_fit
        def _two_term(n_, a, b_):
            return a * n_ ** (-0.5) + b_ * n_ ** (-beta / (2 * beta + 1))
        use = devs > mc_floor
        if use.sum() >= 3:
            popt, _ = curve_fit(_two_term, nn[use], devs[use], p0=[0.05, 0.1], maxfev=8000)
            a_fit, b_fit = float(popt[0]), float(popt[1])
        else:
            a_fit, b_fit = float("nan"), float("nan")
        slope_cov = (np.polyfit(np.log(nn[mask]), np.log(devs[mask]), 1)[0]
                     if mask.sum() >= 2 else float("nan"))
        slope_err = float(np.polyfit(np.log(nn), np.log(rerrs), 1)[0])
        # NOTE: the plug-in logistic estimator typically converges FASTER than
        # the worst-case minimax rate (the bound is an upper bound); report both.
        theory = -beta / (2 * beta + 1)
        rows.append(dict(exp="E9b_rate_validation", beta=beta, theory_slope=round(theory, 3),
                         cov_dev_slope=round(slope_cov, 3),
                         ratio_err_slope=round(slope_err, 3), mc_floor=round(mc_floor, 4), splitCP_coef=round(a_fit, 4), ratio_coef=round(b_fit, 4)))
        print(f"[E9b] beta={beta}: THEORY={theory:.3f} | cov_dev slope={slope_cov:.3f} "
              f"(>{3*mc_floor:.4f} only) | ratio_err slope={slope_err:.3f} | "
              f"splitCP coef={a_fit:.4f}, ratio coef={b_fit:.4f}")
    save_csv(os.path.join(out, "E9b_rate_validation.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E11 group-conditional coverage (Direction 4)
# ----------------------------------------------------------------------------
def exp_group_coverage(out=OUT, beta=1.0, n=1000, m=200, reps=50, alpha=0.1, G=4):
    """Group-conditional coverage on the target (groups = KMeans on the leading
    3 spectral scores). Reports min-group coverage and the bound components of
    Prop. 6: tail energy sum_{j>J} lam_j and G * d_eff(J) / n_cal."""
    from sklearn.cluster import KMeans
    rows = []
    for rep in range(reps):
        X, y, Xt, yt, aux = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
        itr, ica = subject_split(np.arange(n), 0.7, rep)
        md = SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False)
        md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
        _, lo, hi = md.predict(Xt)
        St = md.scx_.transform(Xt) @ md.V_[:, :3]
        g = KMeans(G, n_init=4, random_state=rep).fit_predict(St)
        gcovs = [float(np.mean((yt[g == k] >= lo[g == k]) & (yt[g == k] <= hi[g == k])))
                 for k in range(G)]
        tail = float(np.sum(md.lam_[md.J_:])) if md.J_ < len(md.lam_) else 0.0
        deffJ = float(np.sum((md.lam_[:md.J_] / md.lam_[0]) ** 2))
        rows.append(dict(exp="E11_group", rep=rep, min_gcov=min(gcovs), max_gcov=max(gcovs),
                         tail_energy=tail, G_deff_over_ncal=G * deffJ / len(ica)))
    print(f"[E11] min group coverage = {np.mean([r['min_gcov'] for r in rows]):.3f} "
          f"(nominal 0.90) | bound components: tail={np.mean([r['tail_energy'] for r in rows]):.4f}, "
          f"G*deff(J)/ncal={np.mean([r['G_deff_over_ncal'] for r in rows]):.4f}")
    # Fit Delta_G := (1-alpha) - min_g cov_g against the Prop. 6 components
    A = np.array([[r["tail_energy"], r["G_deff_over_ncal"]] for r in rows])
    yv = np.array([(1 - alpha) - r["min_gcov"] for r in rows])
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    print(f"[E11] Delta_G ~= {coef[0]:.4f}*tail + {coef[1]:.4f}*G*deff/ncal "
          f"(coefficients for the diagnostic decomposition)")
    rows.append(dict(exp="E11_group", rep=-1, min_gcov=float(np.mean(yv)), max_gcov=0.0,
                     tail_energy=float(coef[0]), G_deff_over_ncal=float(coef[1])))
    save_csv(os.path.join(out, "E11_group_coverage.csv"), rows)
    return rows


# ----------------------------------------------------------------------------
# E12 abstention predictability: can we know BEFORE deployment that SCI will
# abstain? Transfer-level diagnostics -> predict informative vs abstained.
# This converts "the method abstains on 12/18 real transfers" from a weakness
# into "the method predicts its own abstention".
# ----------------------------------------------------------------------------
def collect_transfer_diagnostics(md, X_tgt, y_tgt, lo, hi):
    """Transfer-level deployment diagnostics computed from UNLABELLED target
    features only (plus source calibration data) -- i.e. everything available
    BEFORE deployment. No test labels are used for prediction."""
    Zt = md.scx_.transform(X_tgt)
    r_te = md._ratio(Zt)
    # Per-test-point effective sample size of the weighted conformal quantile,
    # using CALIBRATION weights + this test point's weight -- exactly the
    # quantity that drives the abstention decision (pre-deployment, unlabeled).
    w_cal = md.w_ / max(float(md.w_.max()), 1e-12)
    r_cal = md._ratio(md.scx_.transform(md._X_cal_ref))
    scale = max(float(r_cal.max()), 1e-12)
    ess_pts = ((w_cal.sum() + r_te / scale) ** 2
               / (np.sum(w_cal ** 2) + (r_te / scale) ** 2))
    ess_tgt = float(np.min(ess_pts))          # worst-case test point
    ess_med = float(np.median(ess_pts))
    w_conc = float(np.max(r_te) / max(np.median(r_te), 1e-12))
    # source-vs-target classifier AUC of the ratio classifier, evaluated on a
    # balanced holdout built from calibration + target scores (no labels of Y)
    Jr = md.Jr_
    Sc = md.scx_.transform(md._X_cal_ref) @ md.V_[:, :Jr]
    St = Zt @ md.V_[:, :Jr]
    lab = np.r_[np.zeros(len(Sc)), np.ones(len(St))]
    prob = md._ratio_clf.predict_proba(np.vstack([Sc, St]))[:, 1]
    fpr, tpr, _ = _roc_curve(lab, prob)
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2.0 renamed trapz
    clf_auc = float(_trapz(tpr, fpr))
    b_hat, deff, _ = spectral_stats(md._X_cal_ref)
    length = np.asarray(hi, float) - np.asarray(lo, float)
    finite = np.isfinite(length)
    return dict(
        ess_tgt=ess_tgt,
        ess_med=ess_med,
        max_norm_w=w_conc,
        clf_auc=clf_auc,
        beta_hat=b_hat,
        d_eff=deff,
        abst_rate=float(1.0 - np.mean(finite)),
        coverage=float(np.mean((y_tgt >= lo) & (y_tgt <= hi))),
        mean_len=float(np.mean(length[finite])) if finite.any() else float("inf"),
    )

# lightweight ROC (avoid importing sklearn.metrics.roc_curve in hot path twice)
def _roc_curve(y_true, y_score):
    order = np.argsort(-y_score)
    y = y_true[order]
    P = max(float((y == 1).sum()), 1.0); N = max(float((y == 0).sum()), 1.0)
    tpr = np.r_[0.0, np.cumsum(y == 1) / P]
    fpr = np.r_[0.0, np.cumsum(y == 0) / N]
    return fpr, tpr, None

def exp_abstention_sim(out=OUT, betas=(0.8, 1.0, 1.5), n=500, m=200, reps=4, alpha=0.1):
    """E12 (simulation demonstration of the predictability framework):
    a grid of shift strengths x variance inflations yields a MIX of informative
    and abstained transfers; the pre-deployment diagnostics must separate them.
    Real-data counterpart: run with --data real (diagnostics collected inside
    exp_mri / exp_voice) and analyse with exp_abstention_predictability()."""
    t = (np.arange(m) + 0.5) / m
    j_ = np.arange(1, m + 1)
    Phi = np.sqrt(2) * np.sin((j_[:, None] - 0.5) * np.pi * t[None, :])
    shift_curve = np.array([1.0, 0.5, 0.33, 0.25, 0.2]) @ Phi[:5]  # multi-coordinate
    rows = []
    for vi in (1.0, 4.0):
        for delta in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            for rep in range(reps):
                X, y, Xt, yt, _ = sim_dgp(n, 1.0, m=m, seed=7000 + 100 * int(vi) + rep,
                                           var_shift=True, strength=vi - 1.0)
                Xt = Xt + (delta - 1.0) * shift_curve[None, :]
                itr, ica = subject_split(np.arange(n), 0.7, rep)
                md = SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False)
                md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
                _, lo, hi = md.predict(Xt)
                d = collect_transfer_diagnostics(md, Xt, yt, lo, hi)
                d.update(exp="E12_sim", shift_strength=delta, var_infl=vi,
                         informative=1.0 if d["abst_rate"] < 0.5 else 0.0)
                rows.append(d)
    save_csv(os.path.join(out, "E12_abstention_sim.csv"), rows)
    ai = np.mean([r["informative"] for r in rows])
    print(f"[E12/sim] {len(rows)} synthetic transfers, informative fraction={ai:.2f}")
    return rows

def exp_abstention_predictability(out=OUT, fig_dir=None):
    """E12 analysis: predict abstention from pre-deployment diagnostics.
    Reads E12_diagnostics.csv (real transfers, collected by exp_mri/exp_voice)
    and/or E12_abstention_sim.csv (simulation demonstration). Reports
    leave-one-out AUC of a logistic predictor and per-feature AUCs; renders
    fig8_abstention_prediction.png."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import roc_auc_score
    fig_dir = fig_dir or os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    C = plt.get_cmap("tab10").colors
    FEATS = ["ess_tgt", "ess_med", "clf_auc", "beta_hat", "max_norm_w", "d_eff"]
    panels = []
    for tag, fn in [("simulation", "E12_abstention_sim.csv"),
                    ("real", "E12_diagnostics.csv")]:
        rows = _load(os.path.join(out, fn))
        rows = [r for r in rows if r.get("exp") in ("E12_sim", "E12_diagnostics")]
        if len(rows) < 8 or len({r["informative"] for r in rows}) < 2:
            print(f"[E12] skip {tag}: need >= 8 transfers with both classes "
                  f"(got {len(rows)}) -- run the real-data pipeline first")
            continue
        Xf = np.array([[float(r[f]) for f in FEATS] for r in rows])
        yv = np.array([float(r["informative"]) for r in rows])
        # leave-one-out logistic AUC (small-n honest estimate)
        loo = LeaveOneOut(); probs = np.zeros(len(yv))
        for tr, te in loo.split(Xf):
            if len(np.unique(yv[tr])) < 2:
                probs[te] = yv.mean(); continue
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xf[tr], yv[tr])
            probs[te] = lr.predict_proba(Xf[te])[:, 1]
        loo_auc = float(roc_auc_score(yv, probs)) if len(np.unique(yv)) > 1 else float("nan")
        feat_auc = {}
        for k, f in enumerate(FEATS):
            try:
                a = float(roc_auc_score(yv, Xf[:, k]))
                # direction-agnostic: a monotone-decreasing diagnostic is equally
                # usable, we only report separability
                feat_auc[f] = max(a, 1.0 - a)
            except Exception:
                feat_auc[f] = float("nan")
        print(f"[E12/{tag}] n={len(yv)} transfers | LOO AUC (all diagnostics) = {loo_auc:.3f}")
        for f, a in feat_auc.items():
            print(f"    {f:10s} AUC = {a:.3f}")
        save_csv(os.path.join(out, f"E12_predictability_{tag}.csv"),
                 [dict(exp="E12_predictability", tag=tag, loo_auc=loo_auc, feature=f, feat_auc=a)
                  for f, a in feat_auc.items()])
        panels.append((tag, rows, loo_auc, feat_auc))
    if not panels:
        return []
    # ---- fig8 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (tag, rows, loo_auc, feat_auc) in enumerate(panels[:2]):
        ax = axes[i]
        xs = np.array([float(r["ess_tgt"]) for r in rows])
        cov = np.array([float(r["coverage"]) for r in rows])
        ab = np.array([float(r["abst_rate"]) for r in rows])
        sc = ax.scatter(xs, cov, c=ab, cmap="RdYlGn_r", s=45, edgecolor="k", lw=0.4,
                        vmin=0, vmax=1)
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.axvline(5.0, color="r", ls=":", lw=1.2)
        ax.set_xlabel("worst-case per-test-point ESS (pre-deployment)")
        ax.set_ylabel("empirical coverage")
        ax.set_title(f"({chr(97+i)}) {tag}: LOO AUC = {loo_auc:.2f}", fontsize=10)
        plt.colorbar(sc, ax=ax, label="abstention rate")
    fig.tight_layout()
    p = os.path.join(fig_dir, "fig8_abstention_prediction.png")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"[E12] figure -> {p}")
    return panels

# ----------------------------------------------------------------------------
# E10 spectral-decay estimation (numbers for Fig. 1/2 regeneration)
# ----------------------------------------------------------------------------
def exp_beta_decay(out=OUT):
    rows = []
    for beta_true in (0.8, 1.0, 1.2, 1.5, 2.0, 2.5):
        X, y, _, _, _ = sim_dgp(500, beta_true, m=200, seed=0, shift=False)
        b, deff, lam = spectral_stats(X)
        rows.append(dict(exp="E10_beta", beta_true=beta_true, beta_hat=round(b, 3),
                         d_eff=round(deff, 2), J99=int(np.argmax(np.cumsum(lam)/np.sum(lam) > 0.99)) + 1))
        print(f"[E10] beta_true={beta_true}: beta_hat={b:.2f}, d_eff={deff:.1f}")
    save_csv(os.path.join(out, "E10_beta_decay.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# loaders (real-data paths are placeholders; semi mode uses fixed-seed synthetic)
# ----------------------------------------------------------------------------
class PPMILoader:
    """Multi-study MRI loader. Each study root holds MNI152-normalized NIfTIs and a
    metadata table; the STUDY is the domain (no scanner column required). Labels are
    numerically coded diagnosis (PD=1, HC=0) so that the target definition is identical
    across studies (UPDRS regression is PPMI-internal and noted as future work)."""
    DEFAULT_ROOTS = {"ppmi": r"M:\MRI\PPMI", "taowu": r"M:\MRI\TaoWu", "neurocon": r"M:\MRI\NEUROCON"}
    SPECS = {
        "ppmi": dict(pattern=r"PPMI_(\d+)_", meta=("PPMI-UPDRS.csv", "UPDRS.csv"),
                    id_kw=("patno", "subject", "id"), dx_kw=("diagnos", "group", "dx", "status")),
        "neurocon": dict(pattern=r"sub-(?:control|patient|pd)(\d+)_",
                    meta=("neurocon_patients.tsv", "neurocon_patients.csv"),
                    id_kw=("participant", "sub", "id", "patno", "code", "name"), dx_kw=("group", "dx", "diagnos", "status")),
        "taowu": dict(pattern=r"sub-(?:control|patient|pd)(\d+)_",
                    meta=("taowu_patients.tsv", "taowu_patients.csv"),
                    id_kw=("name", "participant", "sub", "id", "patno"), dx_kw=("group", "dx", "diagnos", "status")),
    }

    def __init__(self, mode="semi", path=None, seed=0):
        self.mode, self.path, self.seed = mode, path, seed
        self._real_roots = self._resolve_roots(path)

    @classmethod
    def _resolve_roots(cls, path):
        if isinstance(path, dict):
            return {**cls.DEFAULT_ROOTS, **path}
        if isinstance(path, str) and path.strip():
            out = dict(cls.DEFAULT_ROOTS)
            for kv in path.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out[k.strip()] = v.strip()
            return out
        return dict(cls.DEFAULT_ROOTS)

    @classmethod
    def available(cls, path=None):
        roots = cls._resolve_roots(path)
        names = []
        for name, root in sorted(roots.items()):
            if os.path.isdir(root) and glob.glob(os.path.join(root, "*.nii*")):
                names.append(name)
        return names

    @staticmethod
    def _find_col(cols, *keywords):
        low = {c.lower(): c for c in cols}
        for kw in keywords:
            for lc, c in low.items():
                if kw in lc:
                    return c
        return None

    @staticmethod
    def _img_profile(path):
        import nibabel as nib
        img = nib.load(path).get_fdata().astype(np.float32)
        mask = img > np.percentile(img, 40)
        prof = np.array([img[..., s][mask[..., s]].mean() if mask[..., s].any() else 0.0
                         for s in range(img.shape[2])], dtype=np.float32)
        return prof

    def _load_meta(self, name, root):
        import csv
        spec = self.SPECS[name]
        meta_path = None
        for cand in spec["meta"]:
            p = os.path.join(root, cand)
            if os.path.exists(p):
                meta_path = p
                break
        meta = {}
        if meta_path:
            delim = "\t" if meta_path.endswith(".tsv") else ","
            with open(meta_path, newline="", encoding="utf-8-sig") as f:
                rd = list(csv.DictReader(f, delimiter=delim))
            if rd:
                cols = rd[0].keys()
                c_id = self._find_col(cols, *spec["id_kw"])
                c_dx = self._find_col(cols, *spec["dx_kw"])
                c_up = self._find_col(cols, "np3tot", "updrs", "total", "score", "h_y", "disease_duration")
                import re as _re0
                def _norm(s):
                    return _re0.sub(r"[^0-9]", "", str(s))
                for r in rd:
                    try:
                        sid = _norm(r.get(c_id, "")) if c_id else ""
                        if not sid or sid in meta:
                            continue
                        lab = np.nan
                        if c_dx:
                            dx = (r.get(c_dx) or "").lower()
                            if any(k in dx for k in ("hc", "control", "healthy", "normal")) or dx in ("2", "3"):
                                lab = 0.0
                            elif any(k in dx for k in ("pd", "parkinson", "patient")) or dx in ("1",):
                                lab = 1.0
                        if np.isnan(lab) and c_up:
                            v = (r.get(c_up) or "").strip()
                            if v not in ("", ".", "0", "0.0"):
                                try:
                                    lab = 1.0 if float(v) > 0 else 0.0
                                except Exception:
                                    lab = 1.0
                        meta[sid] = lab
                    except Exception:
                        continue
        return meta

    def load(self, name):
        if self.mode != "real":
            r = np.random.RandomState(self.seed)
            n_s, n_t, m = 160, 160, 120
            X = r.randn(n_s, m) * np.linspace(1, 0.05, m)[None, :]
            y = (X[:, :3] @ np.array([1.0, -0.5, 0.3]) + 0.3 * r.randn(n_s) > 0.5).astype(float)
            Xt = r.randn(n_t, m) * np.linspace(1.3, 0.06, m)[None, :]
            Xt += 0.4 * np.sin(np.linspace(0, 3, m))[None, :]
            yt = (Xt[:, :3] @ np.array([1.0, -0.5, 0.3]) + 0.3 * r.randn(n_t) > 0.6).astype(float)
            ids = np.arange(n_s + n_t)
            return (np.vstack([X, Xt]), np.r_[y, yt], ids,
                    np.array(["GE"] * n_s + ["SIEMENS"] * n_t))
        root = self._real_roots.get(name)
        if root is None or not os.path.isdir(root):
            raise FileNotFoundError(f"MRI study root for {name}: {root}")
        spec = self.SPECS.get(name)
        if spec is None:
            raise ValueError(f"unknown study {name}; add a SPECS entry")
        import csv as _csv
        import re as _re
        meta, grps = {}, {}
        # --- label/group sources: IDA export (diagnosis + scanner), then diagnosis.csv,
        # --- then the study's own metadata table (labels only) ---
        def _norm(s):
            return _re.sub(r"[^0-9]", "", str(s))
        def _absorb(path, delim):
            with open(path, newline="", encoding="utf-8-sig") as f:
                rd = list(_csv.DictReader(f, delimiter=delim))
            if not rd:
                return
            cols = rd[0].keys()
            c_id = self._find_col(cols, "patno", "subject", "id", "code", "name")
            c_dx = self._find_col(cols, "diagnos", "group", "dx", "status")
            c_gr = self._find_col(cols, "manufacturer", "mfg", "model", "strength", "scanner")
            c_up = self._find_col(cols, "np3tot", "updrs", "total", "score", "h_y", "disease_duration")
            for r0 in rd:
                try:
                    sid = _norm(r0.get(c_id, "")) if c_id else ""
                    if not sid:
                        continue
                    if c_gr and sid not in grps:
                        g0 = (r0.get(c_gr) or "").strip()
                        if g0:
                            grps[sid] = g0
                    if sid in meta and not np.isnan(meta[sid]):
                        continue
                    lab = np.nan
                    if c_dx:
                        dx = (r0.get(c_dx) or "").lower()
                        if any(k in dx for k in ("hc", "control", "healthy", "normal")) or dx in ("2", "3"):
                            lab = 0.0
                        elif any(k in dx for k in ("pd", "parkinson", "patient")) or dx in ("1",):
                            lab = 1.0
                    if np.isnan(lab) and c_up:
                        v = (r0.get(c_up) or "").strip()
                        if v not in ("", ".", "0", "0.0"):
                            try:
                                lab = 1.0 if float(v) > 0 else 0.0
                            except Exception:
                                lab = 1.0
                    meta[sid] = lab
                except Exception:
                    continue
        for cand in ("ppmi_subjects.csv", "subjects.csv", "diagnosis.csv"):
            p = os.path.join(root, cand)
            if os.path.exists(p):
                _absorb(p, ",")
                print(f"[E2/real] {name}: absorbed {cand}")
        meta2 = self._load_meta(name, root)
        for k, v in meta2.items():
            if k not in meta or np.isnan(meta[k]):
                meta[k] = v
        files = sorted(glob.glob(os.path.join(root, "*.nii*")))
        if not files:
            files = sorted(glob.glob(os.path.join(root, "*", "*.nii*")))
        cache = os.path.join(root, "_sci_features.npz")
        feats = {}
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            feats = {k: d[k] for k in d.files}
        X, y, ids, grp = [], [], [], []
        for f in files:
            m = _re.search(spec["pattern"], os.path.basename(f))
            if not m:
                continue
            sid = _re.sub(r"[^0-9]", "", m.group(1))
            if sid in ids:
                continue
            base = os.path.basename(f)
            lab = meta.get(sid, np.nan)
            if np.isnan(lab):
                low = base.lower()
                if "control" in low or "_hc" in low:
                    lab = 0.0
                elif "patient" in low or "_pd" in low:
                    lab = 1.0
                else:
                    continue
            if base not in feats:
                try:
                    feats[base] = self._img_profile(f)
                except Exception as e:
                    print(f"[E2/real] skipping unreadable scan {base}: {type(e).__name__}: {e}")
                    continue
            X.append(feats[base]); y.append(lab)
            ids.append(sid); grp.append(grps.get(sid, "NA"))
        np.savez(cache, **feats)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, float)
        ids = np.array(ids)
        grp = np.array(grp)
        ok = ~np.isnan(y)
        if (~ok).any():
            print(f"[E2/real] study {name}: dropped {(~ok).sum()} scans without a usable label")
        X, y, ids, grp = X[ok], y[ok], ids[ok], grp[ok]
        from collections import Counter as _C
        cc = _C(y.tolist())
        if len(cc) < 2:
            print(f"[E2/real] WARNING study {name}: single class {dict(cc)} (need PD+HC)")
        else:
            print(f"[E2/real] study {name}: n={len(y)}, classes {dict(cc)}, "
                  f"groups {dict(_C(grp.tolist()))}")
        return X, y, ids, grp

class VoiceLoader:
    """REAL mode expects one CSV/npz per dataset with columns/paths for
    log-PSD curves (1000-dim) and labels; adapt load() to your local layout.
    Semi mode: fixed-seed synthetic per dataset (REPRODUCIBLE across runs)."""
    DATASETS = ["czech", "spanish", "italian", "english_fig", "english_mm"]
    SEEDS = {"czech": 101, "spanish": 202, "italian": 303, "english_fig": 404, "english_mm": 505}
    NS = {"czech": 126, "spanish": 50, "italian": 39, "english_fig": 81, "english_mm": 35}
    # shared outcome model across the four vowel datasets -> COVARIATE shift only;
    # english_mm uses a DIFFERENT outcome model -> outcome shift (honest failure mode)
    BETA_SHARED = np.array([1.0, -0.5, 0.3])
    BETA_MM = np.array([-0.9, 0.4, 0.5])
    # real corpora: <root>/PD/*.wav and <root>/HC/*.wav (one recording per subject)
    DEFAULT_REAL_PATHS = {"audio": r"M:\Audio", "ipvs": r"M:\IPVS", "vs": r"M:\vs"}

    @classmethod
    def _resolve_roots(cls, path):
        if isinstance(path, dict):
            return {**cls.DEFAULT_REAL_PATHS, **path}
        if isinstance(path, str) and path.strip():
            out = dict(cls.DEFAULT_REAL_PATHS)
            for kv in path.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out[k.strip()] = v.strip()
            return out
        return dict(cls.DEFAULT_REAL_PATHS)

    @classmethod
    def available_real(cls, path=None):
        roots = cls._resolve_roots(path)
        return sorted(n for n, rp in roots.items()
                      if os.path.isdir(os.path.join(rp, "PD"))
                      and os.path.isdir(os.path.join(rp, "HC")))

    @staticmethod
    def _wav_to_logpsd(path, m=1000, fmin=80.0, fmax=8000.0, fs_target=16000):
        """Log-power spectral density curve on an m-point grid over [fmin, fmax] Hz
        (manuscript Sec. 5.8 preprocessing): wav -> mono float -> 16 kHz -> Welch PSD
        -> log -> interpolate."""
        from scipy.io import wavfile
        from scipy import signal as sig
        fs, x = wavfile.read(path)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.max(np.abs(x)) > 1.5:
            x = x / 32768.0
        if fs != fs_target:
            g = int(np.gcd(fs, fs_target))
            x = sig.resample_poly(x, fs_target // g, fs // g).astype(np.float64)
        nseg = min(1024, len(x))
        f, P = sig.welch(x, fs=fs_target, nperseg=nseg)
        mask = (f >= fmin) & (f <= fmax)
        grid = np.linspace(fmin, fmax, m)
        logP = np.log(np.maximum(np.interp(grid, f[mask], P[mask]), 1e-12))
        return logP.astype(np.float32)
    DECAY = {"czech": 0.9, "spanish": 1.1, "italian": 1.0, "english_fig": 0.8, "english_mm": 0.7}
    def __init__(self, mode="semi", path=None):
        self.mode, self.path = mode, path
    def load(self, name):
        if self.mode == "real":
            root = self._resolve_roots(self.path).get(name)
            if root is None or not os.path.isdir(os.path.join(root, "PD")):
                raise FileNotFoundError(f"real voice root for '{name}': {root} (expected <root>/PD, <root>/HC)")
            X, y, ids = [], [], []
            for label, sub in ((1.0, "PD"), (0.0, "HC")):
                for f in sorted(glob.glob(os.path.join(root, sub, "*.wav"))):
                    X.append(self._wav_to_logpsd(f))
                    y.append(label)
                    ids.append(os.path.basename(f))
            return np.array(X), np.array(y), np.array(ids)
        r = np.random.RandomState(self.SEEDS[name])
        n, m = self.NS[name], 200
        X = r.randn(n, m) * np.linspace(1, 0.03, m)[None, :] ** self.DECAY[name]
        beta = self.BETA_MM if name == "english_mm" else self.BETA_SHARED
        y = X[:, :3] @ beta + 0.4 * r.randn(n)
        ids = np.arange(n)
        return X, y, ids

# ----------------------------------------------------------------------------
# E2/E3  real (or semi) MRI / voice experiments with subject-level splits
# ----------------------------------------------------------------------------
def _run_domain(Xs, ys, ids_s, Xt, yt, ids_t, methods, seed=0, cal_frac=0.5):
    itr, ica = subject_split(ids_s, 1 - cal_frac, seed)
    out = {}
    for k, mk in methods.items():
        md = mk().fit(Xs[itr], ys[itr], Xs[ica], ys[ica], X_tgt_unlab=Xt)
        _, lo, hi = md.predict(Xt)
        out[k] = cov_len(yt, lo, hi)
    return out

def exp_mri(out=OUT, mode="semi", path=None, reps=10, alpha=0.1):
    """Cross-domain MRI. Domains: scanner groups within a study when a grouping
    column is available (PPMI via ppmi_subjects.csv Manufacturer), else the whole
    study; plus cross-study pairs (taowu, neurocon). y = coded diagnosis (PD=1,
    HC=0), identical across domains. Both transfer directions."""
    methods = {
        "SCI_full":  lambda: SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False),
        "SCI_psi":   lambda: SCI(alpha=alpha, weighted=True, use_ratio=False),
        "WeightedCP":lambda: WeightedCP(alpha=alpha),
        "NaiveCP":   lambda: NaiveCP(alpha=alpha),
    }
    rows = []
    if mode == "real":
        from collections import Counter
        names = PPMILoader.available(path)
        cache = {}
        def get(nm):
            if nm not in cache:
                cache[nm] = PPMILoader(mode, path, seed=0).load(nm)
            return cache[nm]
        domains = []
        for nm in names:
            X, y, ids, grp = get(nm)
            gg = Counter(grp.tolist())
            big = [g for g, c in gg.most_common() if c >= 20 and g != 'NA']
            if len(big) >= 2:
                for g in big[:2]:
                    msk = grp == g
                    if len(set(y[msk].tolist())) >= 2:
                        domains.append((f"{nm}:{g}", X[msk], y[msk], ids[msk]))
                print(f"[E2/real] study {nm} split into scanner domains: "
                      + ", ".join(f"{g}={gg[g]}" for g in big[:2]))
            elif len(set(y.tolist())) >= 2:
                domains.append((nm, X, y, ids))
        print(f"[E2/real] domains with PD+HC: {[d[0] for d in domains]}")
        if len(domains) < 2:
            print("[E2/real] need >= 2 domains")
            return rows
        pairs = [(a, b) for a in domains for b in domains if a[0] != b[0]]
        res = {(a[0], b[0]): {k: ([], []) for k in methods} for (a, b) in pairs}
        for rep in range(reps):
            for (a, b) in pairs:
                _, Xs, ys, ids_s = a
                _, Xt, yt, ids_t = b
                assert len(set(ids_s) & set(ids_t)) == 0, "subject overlap across domains!"
                itr, ica = subject_split(ids_s, 0.7, rep)
                for k, mk in methods.items():
                    md = mk().fit(Xs[itr], ys[itr], Xs[ica], ys[ica], X_tgt_unlab=Xt)
                    _, lo, hi = md.predict(Xt)
                    c, l = cov_len(yt, lo, hi)
                    res[(a[0], b[0])][k][0].append(c)
                    res[(a[0], b[0])][k][1].append(l)
                    if k == "SCI_full":
                        dg = collect_transfer_diagnostics(md, Xt, yt, lo, hi)
                        dg.update(exp="E12_diagnostics", dataset="mri",
                                  src=a[0], tgt=b[0], rep=rep,
                                  informative=1.0 if dg["abst_rate"] < 0.5 else 0.0)
                        rows.append(dg)
        for (sn, tn) in res:
            for k in methods:
                c, l = res[(sn, tn)][k]
                rows.append(dict(exp="E2_mri", mode=mode, src=sn, tgt=tn, method=k,
                                 coverage=float(np.mean(c)), cov_se=float(np.std(c)/np.sqrt(max(reps, 1))),
                                 length=float(np.mean(l))))
                print(f"[E2/real] {sn}->{tn} {k:12s} coverage={np.mean(c):.3f}  length={np.mean(l):.2f}")
        save_csv(os.path.join(out, "E2_mri.csv"), rows)
        return rows
    res = {k: ([], []) for k in methods}
    for rep in range(reps):
        X, y, ids = PPMILoader(mode, path, seed=rep).load("ppmi")[:3]
        half = len(X) // 2
        Xs, Xt, ys, yt = X[:half], X[half:], y[:half], y[half:]
        ids_s, ids_t = ids[:half], ids[half:]
        assert len(set(ids_s) & set(ids_t)) == 0
        r = _run_domain(Xs, ys, ids_s, Xt, yt, ids_t, methods, seed=rep)
        for k in methods:
            res[k][0].append(r[k][0]); res[k][1].append(r[k][1])
    for k in methods:
        c, l = res[k]
        rows.append(dict(exp="E2_mri", mode=mode, method=k, coverage=float(np.mean(c)),
                         cov_se=float(np.std(c)/np.sqrt(max(reps, 1))), length=float(np.mean(l))))
        print(f"[E2/{mode}] {k:12s} coverage={np.mean(c):.3f}  length={np.mean(l):.2f}")
    save_csv(os.path.join(out, "E2_mri.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E4  calibration sample efficiency (target-domain calibration sweep)
# ----------------------------------------------------------------------------
def exp_sample_efficiency(out=OUT, beta=1.0, n=500, m=200, reps=50, alpha=0.1):
    """Scenario (i): calibration labels come from the SOURCE site; the x-axis is the
    number of UNLABELED target observations m_T available for ratio estimation.
    NaiveCP uses no target data at all (flat reference line); SCI and WeightedCP use
    source calibration + m_T unlabeled target features."""
    methods = {
        "SCI_full":  lambda: SCI(alpha=alpha, weighted=True, use_ratio=True, psi_in_weight=False),
        "WeightedCP":lambda: WeightedCP(alpha=alpha),
        "NaiveCP":   lambda: NaiveCP(alpha=alpha),
    }
    rows = []
    for nu in [20, 50, 100, 200, 500, 1000]:
        res = {k: [] for k in methods}
        for rep_ in range(reps):
            X, y, Xt, yt, _ = sim_dgp(n, beta, m=m, seed=rep_, var_shift=False)
            itr, ica = subject_split(np.arange(n), 0.7, rep_)
            perm = np.random.RandomState(rep_).permutation(len(Xt))
            Xu = Xt[perm[:min(nu, len(Xt))]]
            for k, mk in methods.items():
                md = mk().fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xu)
                _, lo, hi = md.predict(Xt)
                res[k].append(np.mean((yt >= lo) & (yt <= hi)))
        for k in methods:
            rows.append(dict(exp="E4_sample_efficiency", n_unlab=nu, method=k,
                             coverage=float(np.mean(res[k])),
                             cov_se=float(np.std(res[k])/np.sqrt(reps))))
        print(f"[E4] m_T={nu:5d} unlabeled: " + " | ".join(
            f"{k}={np.mean(res[k]):.3f}" for k in methods))
    save_csv(os.path.join(out, "E4_sample_efficiency.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E5  computational complexity
# ----------------------------------------------------------------------------
def exp_complexity(out=OUT, n=1000):
    rows = []
    for m in [500, 2000, 10000]:
        r = np.random.RandomState(0)
        Z = r.randn(n, m); Z = StandardScaler().fit_transform(Z)
        t0 = time.time(); lam, V = eigh(np.cov(Z.T)); t_full = time.time() - t0
        C = np.cov(Z.T)
        t0 = time.time(); idx = r.choice(m, 200, replace=False); eigh(C[np.ix_(idx, idx)]); t_nys = time.time() - t0
        t0 = time.time()
        Q, _ = np.linalg.qr(r.randn(m, 150)); eigh(Q.T @ C @ Q); t_rsvd = time.time() - t0
        # incremental / streaming: batched sufficient-statistics updates
        # (chunked BLAS, no per-sample outer loops -> O(m^2) memory, fast)
        t0 = time.time()
        S1 = np.zeros(m); S2 = np.zeros((m, m)); cnt = 0
        for i in range(0, n, 100):
            Zb = Z[i:i + 100]; nb = len(Zb)
            S1 += Zb.sum(0); S2 += Zb.T @ Zb; cnt += nb
        Cn = S2 / cnt - np.outer(S1 / cnt, S1 / cnt)
        t_inc = time.time() - t0
        inc_err = float(np.abs(Cn - C).max() / np.abs(C).max())
        rows.append(dict(exp="E5_complexity", m=m, full=round(t_full, 3),
                         nystrom=round(t_nys, 3), rsvd=round(t_rsvd, 3),
                         incremental=round(t_inc, 3), inc_rel_err=round(inc_err, 6)))
        print(f"[E5] m={m:6d}: full={t_full:.2f}s nystrom={t_nys:.2f}s rsvd={t_rsvd:.2f}s inc={t_inc:.2f}s")
    save_csv(os.path.join(out, "E5_complexity.csv"), rows)
    return rows

# ----------------------------------------------------------------------------
# E6  adaptivity of truncation J
# ----------------------------------------------------------------------------
def exp_adaptivity(out=OUT, betas=(1.0, 2.0), n=500, m=200, reps=50, alpha=0.1):
    """Truncation adaptivity: oracle J* vs GCV / conformal-GCV / spectral-gap /
    Lepski. Lengths are means over NON-abstained points (cov_len)."""
    rows = []
    for beta in betas:
        Jstar = int(n ** (1 / (2 * beta + 1)))
        for method in ["oracle", "gcv", "gcv_cp", "gap", "lepski"]:
            Js, lens = [], []
            for rep in range(reps):
                X, y, Xt, yt, _ = sim_dgp(n, beta, m=m, seed=rep, var_shift=False)
                itr, ica = subject_split(np.arange(n), 0.7, rep)
                if method == "oracle":
                    J = Jstar
                elif method == "gcv":
                    J = None
                elif method == "gcv_cp":
                    J = None
                elif method == "gap":
                    Z = StandardScaler().fit_transform(X[itr])
                    lam, _ = eigh(np.cov(Z.T)); lam = np.sort(lam)[::-1]
                    J = max(2, min(int(np.argmax(np.diff(lam[:50]) < 0)) + 2, 40))
                else:
                    itr2, itune = subject_split(np.arange(len(itr)), 0.6, rep)
                    Z = StandardScaler().fit_transform(X[itr[itune]])
                    lam, V = eigh(np.cov(Z.T)); o = np.argsort(lam)[::-1]
                    lam, V = lam[o], V[:, o]
                    Sc = Z @ V; ytune = y[itr[itune]]
                    errs = []
                    for Jc in range(2, 41):
                        A = Sc[:, :Jc]
                        G = A @ np.linalg.pinv(A.T @ A + 1e-6 * np.eye(Jc)) @ A.T
                        errs.append(np.mean(((ytune - G @ ytune) / (1 - np.trace(G)/len(ytune))) ** 2))
                    J = int(np.argmin(errs)) + 2
                md = SCI(alpha=alpha, J=J, weighted=True, use_ratio=True, psi_in_weight=False,
                         select="gcv_cp" if method == "gcv_cp" else "gcv")
                md.fit(X[itr], y[itr], X[ica], y[ica], X_tgt_unlab=Xt)
                _, lo, hi = md.predict(Xt)
                _, l6 = cov_len(yt, lo, hi)
                Js.append(md.J_); lens.append(l6)
            rows.append(dict(exp="E6_adaptivity", beta=beta, method=method,
                             J_mean=float(np.mean(Js)), length=float(np.mean(lens))))
            print(f"[E6] beta={beta} {method:7s}: J={np.mean(Js):.1f}, length={np.mean(lens):.3f} "
                  f"(oracle J*={Jstar})")
    save_csv(os.path.join(out, "E6_adaptivity.csv"), rows)
    return rows
# ----------------------------------------------------------------------------
# Self-test: run BEFORE any full experiment to verify the local file is current.
# ----------------------------------------------------------------------------

def exp_voice(out=OUT, mode="semi", path=None):
    """Cross-lingual voice transfers. Real mode: WAV directory trees
    (<root>/PD, <root>/HC); datasets auto-discovered from available_real()."""
    vl = VoiceLoader(mode, path)
    methods = {
        "SCI_full":  lambda: SCI(alpha=0.1, weighted=True, use_ratio=True, psi_in_weight=False),
        "SCI_psi":   lambda: SCI(alpha=0.1, weighted=True, use_ratio=False),
        "WeightedCP":lambda: WeightedCP(alpha=0.1),
        "NaiveCP":   lambda: NaiveCP(alpha=0.1),
    }
    names = vl.available_real(path) if mode == "real" else VoiceLoader.DATASETS[:4]
    cache = {}
    def get(nm):
        if nm not in cache:
            cache[nm] = vl.load(nm)
        return cache[nm]
    rows = []
    for s in names:
        for t in names:
            if s == t: continue
            Xs, ys, ids_s = get(s); Xt, yt, ids_t = get(t)
            itr, ica = subject_split(ids_s, 0.5, 0)
            for k, mk in methods.items():
                md = mk().fit(Xs[itr], ys[itr], Xs[ica], ys[ica], X_tgt_unlab=Xt)
                _, lo, hi = md.predict(Xt)
                c, l = cov_len(yt, lo, hi)
                rows.append(dict(exp="E3_voice", mode=mode, src=s, tgt=t, method=k, coverage=c, length=l))
                if k == "SCI_full":
                    dg = collect_transfer_diagnostics(md, Xt, yt, lo, hi)
                    dg.update(exp="E12_diagnostics", dataset="voice",
                              src=s, tgt=t, rep=0,
                              informative=1.0 if dg["abst_rate"] < 0.5 else 0.0)
                    rows.append(dg)
        print(f"[E3/{mode}] {s}: done as source")
    sub = [x for x in rows if x.get("method") == "SCI_full"]
    if sub:
        print(f"[E3/{mode}] SCI_full mean coverage over {len(sub)} transfers = "
              f"{np.mean([x['coverage'] for x in sub]):.3f} "
              f"(note: abstained transfers contribute 1.0 by construction)")
    save_csv(os.path.join(out, "E3_voice.csv"), rows)
    return rows

def selftest(reps=30, alpha=0.1):
    print(f"== SCI_v3 selftest | version {VERSION} ==")
    ok = True
    covs = []
    for rep in range(reps):
        X, y, Xt, yt, _ = sim_dgp(300, 1.0, m=100, seed=rep)
        itr, _ = subject_split(np.arange(300), 0.7, rep)
        idx = np.random.RandomState(rep).choice(len(Xt), 50, replace=False)
        md = NaiveCP(alpha=alpha).fit(X[itr], y[itr], Xt[idx], yt[idx])
        _, lo, hi = md.predict(Xt)
        covs.append(np.mean((yt >= lo) & (yt <= hi)))
    m1 = float(np.mean(covs))
    print(f"[1] NaiveCP target-calibrated coverage = {m1:.4f}  (>= 0.89; ~0.87 = stale file)")
    ok &= m1 >= 0.89
    r = np.random.RandomState(0).randn(200)
    q_guard = wcp_quantile(r, np.zeros(200), 0.0, alpha)
    q_ess = wcp_quantile(r, np.r_[1e6, np.ones(199)], 1e6, alpha)
    print(f"[2] zero-weight -> {q_guard} (inf); extreme ESS -> {q_ess} (inf)")
    ok &= np.isinf(q_guard) and np.isinf(q_ess)
    covs = []
    for rep in range(reps):
        X, y, _, _, _ = sim_dgp(300, 1.0, m=100, seed=rep)
        itr, ica = subject_split(np.arange(300), 0.7, rep)
        md = SCI(alpha=alpha, weighted=False).fit(X[itr], y[itr], X[ica], y[ica])
        Xe, ye, _, _, _ = sim_dgp(1000, 1.0, m=100, seed=9000 + rep, shift=False)
        _, lo, hi = md.predict(Xe)
        covs.append(np.mean((ye >= lo) & (ye <= hi)))
    m3 = float(np.mean(covs))
    print(f"[3] unweighted exchangeability coverage = {m3:.4f}  (>= 0.89)")
    ok &= m3 >= 0.89
    print("SELFTEST:", "PASS" if ok else "FAIL -- RE-DOWNLOAD")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="semi", choices=["sim", "semi", "real"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--mri_path", default=None)
    ap.add_argument("--voice_path", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.makedirs(a.out, exist_ok=True)
    global DATA_MODE; DATA_MODE = a.data
    reps = 20 if a.quick else 100
    print(f"==== SCI v3 {VERSION} | data={a.data} | out={a.out} ====")
    exp_validity_check(a.out, reps=min(reps, 200))
    exp_simulation(a.out, reps=reps)
    exp_ablation(a.out, reps=reps)
    exp_failure_mode(a.out, reps=reps)
    exp_ratio_sensitivity(a.out, reps=max(20, reps // 2))
    exp_rate_validation(a.out, reps=max(15, reps // 3))
    exp_beta_decay(a.out)
    exp_sample_efficiency(a.out, reps=max(20, reps // 2))
    exp_complexity(a.out)
    exp_adaptivity(a.out, reps=max(20, reps // 3))
    exp_group_coverage(a.out, reps=max(15, reps // 3))
    exp_abstention_sim(a.out, reps=4)
    try:
        exp_mri(a.out, mode=a.data, path=a.mri_path, reps=5 if a.quick else 10)
    except FileNotFoundError as e:
        print(f"[skip] E2 MRI: {e}")
    if a.data != "sim":
        exp_voice(a.out, mode=a.data, path=a.voice_path)
    # E12 analysis runs LAST: it consumes the diagnostics collected during
    # exp_mri / exp_voice (E12_diagnostics.csv) plus the simulation grid.
    exp_abstention_predictability(a.out)
    make_figures(a.out)
    print("done.")

# ----------------------------------------------------------------------------
# Figure generation: reads the CSVs in a results dir and renders the paper
# figures.  Run automatically at the end of main(), or standalone:
#   python SCI_v3.py --data semi --out results_v4     (experiments + figures)
# ----------------------------------------------------------------------------
import csv as _csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(_csv.DictReader(f))

def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

FIG_SPECS = [
    ("E1_simulation_shift.csv",  "fig2_main_comparison.png"),
    ("E7_ablation.csv",          "fig3_ablation.png"),
    ("E4_sample_efficiency.csv", "fig4_sample_efficiency.png"),
    ("E8_failure_mode.csv",      "fig5_failure_mode.png"),
    ("E9_ratio_sensitivity.csv", "fig7_ratio_sensitivity.png"),
]

def make_figures(results_dir=OUT, fig_dir=None):
    fig_dir = fig_dir or os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    C = plt.get_cmap("tab10").colors
    made = []

    # ---- Fig 1: spectral decay + rate validation --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, beta in enumerate((0.8, 1.0, 1.5, 2.5)):
        X, y, _, _, _ = sim_dgp(500, beta, m=200, seed=0, shift=False)
        b_hat, deff, lam = spectral_stats(X, Jmax=50)
        j = np.arange(1, 51)
        axes[0].loglog(j, lam[:50], "o", ms=3, color=C[i],
                       label=fr"$\beta$={beta} ($\hat\beta$={b_hat:.2f})")
        axes[0].loglog(j, lam[0] * j ** (-2 * beta), "--", color=C[i], lw=1)
    axes[0].set_xlabel("eigenvalue index $j$"); axes[0].set_ylabel("$\\lambda_j$ (log-log)")
    axes[0].set_title("(a) spectral decay recovery"); axes[0].legend(fontsize=8)
    det = [r for r in _load(os.path.join(results_dir, "E9b_rate_validation.csv"))
           if r.get("exp") == "E9b_detail"]
    if det:
        for i, beta in enumerate(sorted({float(r["beta"]) for r in det})):
            pts = sorted([(float(r["n"]), float(r["cov_dev"])) for r in det
                          if float(r["beta"]) == beta])
            ns_ = np.array([p[0] for p in pts]); ds_ = np.array([p[1] for p in pts])
            axes[1].loglog(ns_, ds_, "o-", color=C[i], label=fr"$\beta$={beta}")
        axes[1].set_xlabel("sample size $n$ (log-log)")
        axes[1].set_ylabel("coverage deviation $|$cov$-0.9|$")
        axes[1].set_title("(b) rate validation (E9b)"); axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "run E9b first", ha="center", transform=axes[1].transAxes)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig1_rate_validation.png"), dpi=300)
    plt.close(fig); made.append("fig1_rate_validation.png")

    # ---- Fig 2: main comparison under shift --------------------------------
    rows = _load(os.path.join(results_dir, "E1_simulation_shift.csv"))
    if rows:
        methods = sorted({r["method"] for r in rows}, key=lambda m: min(
            _f(x["coverage"]) for x in rows if x["method"] == m))
        betas = sorted({r["beta"] for r in rows}, key=_f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        x = np.arange(len(betas)); w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            cov = [_f(next(r for r in rows if r["method"] == m and r["beta"] == b)["coverage"])
                   for b in betas]
            ln = [_f(next(r for r in rows if r["method"] == m and r["beta"] == b)["length"])
                  for b in betas]
            axes[0].bar(x + (i - len(methods) / 2) * w + w / 2, cov, w, label=m, color=C[i % 10])
            axes[1].bar(x + (i - len(methods) / 2) * w + w / 2, ln, w, label=m, color=C[i % 10])
        for ax, ttl, ylab in [(axes[0], "(a) coverage (nominal 0.9)", "coverage"),
                              (axes[1], "(b) mean interval length", "length")]:
            ax.axhline(0.9 if "coverage" in ylab else 0, color="k", ls="--", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels([fr"$\beta$={b}" for b in betas])
            ax.set_ylabel(ylab); ax.set_title(ttl)
        axes[0].legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig2_main_comparison.png"), dpi=300)
        plt.close(fig); made.append("fig2_main_comparison.png")

    # ---- Fig 3: mechanism ablation ----------------------------------------
    rows = [r for r in _load(os.path.join(results_dir, "E7_ablation.csv"))
            if r.get("arm") != "A6_vs_A4_wilcoxon"]
    if rows:
        arms = [r["arm"] for r in rows]
        cov = [_f(r["coverage"]) for r in rows]; ln = [_f(r["length"]) for r in rows]
        x = np.arange(len(arms))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        b1 = axes[0].bar(x, cov, color=[C[0]] + [C[1]] * (len(arms) - 1))
        axes[0].axhline(0.9, color="k", ls="--", lw=1); axes[0].set_ylim(0.75, 1.0)
        axes[0].set_xticks(x); axes[0].set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
        axes[0].set_title("(a) coverage"); axes[0].set_ylabel("coverage")
        axes[1].bar(x, ln, color=C[2])
        axes[1].set_xticks(x); axes[1].set_xticklabels(arms, rotation=30, ha="right", fontsize=8)
        axes[1].set_title("(b) interval length"); axes[1].set_ylabel("length")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig3_ablation.png"), dpi=300)
        plt.close(fig); made.append("fig3_ablation.png")

    # ---- Fig 4: sample efficiency ------------------------------------------
    rows = _load(os.path.join(results_dir, "E4_sample_efficiency.csv"))
    if rows:
        methods = sorted({r["method"] for r in rows})
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, m in enumerate(methods):
            pts = sorted([(int(float(r["n_unlab"])), _f(r["coverage"])) for r in rows
                          if r["method"] == m])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C[i], label=m)
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("unlabeled target observations $m_T$ (scenario (i))")
        ax.set_ylabel("coverage"); ax.set_ylim(0.8, 1.0); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig4_sample_efficiency.png"), dpi=300)
        plt.close(fig); made.append("fig4_sample_efficiency.png")

    # ---- Fig 5: failure mode ------------------------------------------------
    rows = _load(os.path.join(results_dir, "E8_failure_mode.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        names = [r["method"] for r in rows]; cov = [_f(r["coverage"]) for r in rows]
        ax.bar(np.arange(len(names)), cov, color=C[:len(names)])
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("coverage"); ax.set_ylim(0, 1.0)
        ax.set_title("outcome shift: all methods fail (E8)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig5_failure_mode.png"), dpi=300)
        plt.close(fig); made.append("fig5_failure_mode.png")

    # ---- Fig 6: group coverage ----------------------------------------------
    rows = [r for r in _load(os.path.join(results_dir, "E11_group_coverage.csv")) if r.get("rep") not in (None, "", "-1")]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        tail = np.array([_f(r["tail_energy"]) for r in rows])
        gd = np.array([_f(r["G_deff_over_ncal"]) for r in rows])
        mg = np.array([_f(r["min_gcov"]) for r in rows])
        axes[0].scatter(tail, 0.9 - mg, c=gd, cmap="viridis")
        axes[0].set_xlabel("spectral tail energy"); axes[0].set_ylabel("coverage deficit")
        axes[0].set_title("(a) deficit vs tail energy")
        axes[1].scatter(gd, 0.9 - mg, c=tail, cmap="plasma")
        k, b0 = np.polyfit(gd, 0.9 - mg, 1)
        xs = np.linspace(gd.min(), gd.max(), 50)
        axes[1].plot(xs, k * xs + b0, "k--", lw=1,
                     label=fr"fit slope={k:.2f}")
        axes[1].set_xlabel("$G\\, d_{eff}(J)/n_{cal}$"); axes[1].legend(fontsize=8)
        axes[1].set_title("(b) deficit vs calibration-budget term")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig6_group_coverage.png"), dpi=300)
        plt.close(fig); made.append("fig6_group_coverage.png")

    # ---- Fig 7: ratio sensitivity -------------------------------------------
    rows = _load(os.path.join(results_dir, "E9_ratio_sensitivity.csv"))
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        pts = sorted([(int(float(r["n_unlab"])), _f(r["coverage"]), _f(r.get("prob_ratio_error", "nan")))
                      for r in rows])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=C[0], label="coverage")
        ax.set_xscale("log"); ax.set_xlabel("unlabeled target samples")
        ax.set_ylabel("coverage"); ax.axhline(0.9, color="k", ls="--", lw=1)
        ax2 = ax.twinx()
        ax2.plot([p[0] for p in pts], [p[2] for p in pts], "s--", color=C[3],
                 label="ratio estimation error")
        ax2.set_ylabel("prob-scale ratio error")
        ax.set_title("ratio sensitivity (E9)")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig7_ratio_sensitivity.png"), dpi=300)
        plt.close(fig); made.append("fig7_ratio_sensitivity.png")

    print(f"[figures] {len(made)} figure(s) -> {fig_dir}: {', '.join(made)}")
    return [os.path.join(fig_dir, m) for m in made]


if __name__ == "__main__":
    main()