import numpy as np
from scipy.optimize import minimize, root_scalar
from dataclasses import dataclass
from scipy.stats import norm
from black_scholes import calculate_IV

"""
commonly used variables (float or ndarray):
- S: spot price
- K: strike price
- T: maturity in years
- r: risk-free rate
- q: dividend yield
- sigma: volatility
"""

# ❶  Basic Black‑Scholes helpers
#------------------------------------------------------------------------------
SQRT_2PI = np.sqrt(2.0 * np.pi)

def _bs_d1(S, K, T, r, q, sigma):
    if sigma * np.sqrt(T) == 0.0:
        return np.sign(np.log(S/K)) * np.inf
    return (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))

def _bs_delta(S, K, T, r, q, sigma, call=True):
    d1 = _bs_d1(S, K, T, r, q, sigma)
    if call:
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)

# ❷  SSVI slice formulas (per‑maturity)
#------------------------------------------------------------------------------
# def _phi(theta, gamma):
#     """ϕ(θ) backbone."""
#     return (1.0 - (1.0 - np.exp(-gamma * theta)) / (gamma * theta)) / (gamma * theta)
def _phi(theta, gamma):
    return (1.0 - np.exp(-gamma * theta)) / (gamma * theta)

def _ssvi_total_variance(k, theta, gamma, rho):
    """w(k,θ) total variance."""
    p = _phi(theta, gamma)
    return 0.5 * theta * (
        1.0 + rho * p * k + np.sqrt((p * k + rho) ** 2 + 1.0 - rho ** 2)
    )

# ❸  Parametric term‑structure
#------------------------------------------------------------------------------
def theta_param(t, a0, a1, a2):
    """Increasing, positive ATM total variance curve."""
    return a0 * t + a1 * (1.0 - np.exp(-a2 * t))

def rho_param(t, b0, b1):
    """ρ(t) bounded in (‑1,1) via tanh."""
    return np.tanh(b0 + b1 * t)

# ❹  Main class
#------------------------------------------------------------------------------

@dataclass
class eSSVI:
    """
    Smooth (e)SSVI implied volatility surface with simple term-structure.
    Usage:
    1. Initialize with spot price and interest rate, dividend yield, and call or put:
       surf = eSSVI(spot=100.0, r=0.05, q=0.0, call=True)
    2. Calibrate to market option data:
       surf.fit(strikes, maturities, prices, weights=None) 
       #maturities is expiry in years, prices are contract quote prices. all are 1-d float arrays. 
       #weights is optional, used to weigh each contract, for weights, try vega**2 / (spread*T*IV)**2
    3. Query the fitted surface for implied volatility:
       iv = surf.iv(strike=105.0, t=0.027)
       #or
       iv, strike = surf.iv_from_delta(delta_target=0.25, t=0.027)
    """
    spot: float
    call: bool = True
    r: float = 0.04
    q: float = 0.0
    params: np.ndarray = None  # calibrated [a0,a1,a2,b0,b1,gamma]

    # ------------------------------------------------------------------ #
    #  Fitting                                                           #
    # ------------------------------------------------------------------ #
    def fit(self, strikes, maturities, prices, weights=None,
            init_guess=None, bounds=None, penalty_scale=1e3):
        """
        Calibrate the six-parameter surface to (K,T,price) data.
        strikes, maturities, prices, weights are 1-D float np.ndarrays
        note: for weights, try vega**2 / (spread*T*IV)**2
        """
        K = np.asarray(strikes, dtype=float).ravel()
        T = np.asarray(maturities, dtype=float).ravel()
        P = np.asarray(prices, dtype=float).ravel()
        n = len(P)
        if not (len(K) == len(T) == n):
            raise ValueError("strikes, maturities, prices must have same length")

        # market IVs -> total variances
        sig_market = np.array([
            calculate_IV(P[i], self.spot, K[i], T[i],
                         r=self.r, q=self.q,
                         call_or_put="call" if self.call else "put")
            for i in range(n)
        ], dtype=float)
        w_market = sig_market**2 * T

        # forward & log-moneyness
        F = self.spot * np.exp((self.r - self.q) * T)
        k = np.log(K / F)

        # default weights
        if weights is None:
            weights = np.ones_like(P, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float).ravel()

        # filter invalids / tiny maturities
        valid = np.isfinite(w_market) & np.isfinite(k) & (T > 1e-8) & np.isfinite(weights)
        if not np.any(valid):
            raise ValueError("No valid data points after filtering.")
        K, T, P, F, k, w_market, weights = (
            K[valid], T[valid], P[valid], F[valid], k[valid], w_market[valid], weights[valid]
        )

        # sort by maturity so calendar penalties make sense
        order = np.argsort(T)
        K, T, P, F, k, w_market, weights = (
            K[order], T[order], P[order], F[order], k[order], w_market[order], weights[order]
        )

        # init guess
        if init_guess is None:
            a0 = max(0.05, 0.2 * np.nanmean(w_market) / max(np.nanmean(T), 1e-8))
            init_guess = np.array([a0, 0.1, 1.0, 0.0, 0.0, 0.8])  # gamma ~ 0.8 to start

        # bounds (a0,a1,a2,b0,b1,gamma)
        # keep a0,a1,a2 positive to make theta increasing; set gamma >= 0.5 to respect wide-wing limits gracefully
        if bounds is None:
            bounds = [(1e-5,  5.0),     # a0
                      (1e-5,  5.0),     # a1
                      (1e-3, 10.0),     # a2
                      (-5.0,  5.0),     # b0
                      (-5.0,  5.0),     # b1
                      (0.5,  10.0)]     # gamma

        def theta_param(t, a0, a1, a2):
            return a0 * t + a1 * (1.0 - np.exp(-a2 * t))

        def rho_param(t, b0, b1):
            return np.tanh(b0 + b1 * t)

        def objective(theta_rho_gamma):
            a0, a1, a2, b0, b1, gamma = theta_rho_gamma

            # per-observation θ and ρ
            theta = theta_param(T, a0, a1, a2)                        # shape (m,)
            rho   = rho_param(T, b0, b1)                              # shape (m,)

            # model total variance at observed k,T
            w_model = _ssvi_total_variance(k, theta, gamma, rho)

            # === SSVI no-butterfly penalties ===
            # ψ = θ ϕ(θ)
            phi = _phi(theta, gamma)
            psi = theta * phi
            # (1+|ρ|) ψ ≤ 4
            pen_bfly1 = np.maximum(psi * (1.0 + np.abs(rho)) - 4.0, 0.0)
            # (1+|ρ|) θ ϕ(θ)^2 ≤ 4
            pen_bfly2 = np.maximum(theta * (phi**2) * (1.0 + np.abs(rho)) - 4.0, 0.0)

            # === eSSVI no-calendar penalties ===
            # Require ψ non-decreasing with T
            dpsi = np.diff(psi)
            pen_cal_mono = np.minimum(dpsi, 0.0)  # negative diffs violate

            # Cross-slice bound: |ρ_j ψ_j − ρ_i ψ_i| ≤ ψ_j − ψ_i  (only meaningful when dpsi>0)
            num = np.diff(rho * psi)
            den = np.maximum(dpsi, 0.0)
            pen_cal_cross = np.maximum(np.abs(num) - den, 0.0)

            pen = (penalty_scale * (np.sum(pen_bfly1**2) + np.sum(pen_bfly2**2))
                   + penalty_scale * (np.sum(pen_cal_mono**2) + np.sum(pen_cal_cross**2)))

            # data misfit in total variance space
            err = w_model - w_market
            return np.sum(weights * err * err) + pen

        res = minimize(objective, init_guess, bounds=bounds, method="L-BFGS-B")
        if not res.success:
            raise RuntimeError("Calibration failed: " + res.message)
        self.params = res.x
        return res

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _theta(self, t):
        if self.params is None:
            raise RuntimeError("Surface not calibrated yet.")
        a0, a1, a2, *_ = self.params
        return a0 * t + a1 * (1.0 - np.exp(-a2 * t))

    def _rho(self, t):
        if self.params is None:
            raise RuntimeError("Surface not calibrated yet.")
        *_, b0, b1, _ = self.params
        return np.tanh(b0 + b1 * t)

    def _gamma(self):
        if self.params is None:
            raise RuntimeError("Surface not calibrated yet.")
        return float(self.params[-1])

    def total_variance(self, k, t):
        theta = self._theta(np.asarray(t, dtype=float))
        rho   = self._rho(np.asarray(t, dtype=float))
        return _ssvi_total_variance(np.asarray(k, dtype=float), theta, self._gamma(), rho)

    # ------------------------------------------------------------------ #
    #  IV & Delta inversion                                              #
    # ------------------------------------------------------------------ #
    def iv(self, strike, t):
        """Return Black-Scholes implied volatility σ_bs(K,t)."""
        t = float(t)
        if t <= 0.0:
            raise ValueError("t must be > 0 for implied volatility.")
        F = self.spot * np.exp((self.r - self.q) * t)
        k = np.log(float(strike) / F)
        w = self.total_variance(k, t)
        w = np.maximum(w, 0.0)
        return np.sqrt(w / t)

    def iv_from_delta(self, delta_target, t, tol=1e-6, max_iter=100):
        """
        Solve for (σ, K) such that BS delta equals `delta_target` at maturity t.
        Accepts signed delta in (-1,1). E.g., 25Δ put => -0.25; 25Δ call => +0.25.
        """
        if not (-1.0 < delta_target < 1.0):
            raise ValueError("Delta must be in (-1,1).")
        t = float(t)
        if t <= 0.0:
            raise ValueError("t must be > 0.")
        F = self.spot * np.exp((self.r - self.q) * t)

        def f_of_k(k):
            K     = F * np.exp(k)
            sigma = self.iv(K, t)
            dlt   = _bs_delta(self.spot, K, t, self.r, self.q, sigma, call=self.call)
            return dlt - delta_target

        # adaptive symmetric bracketing in log-moneyness
        k_lo, k_hi = -2.5, 2.5
        y_lo, y_hi = f_of_k(k_lo), f_of_k(k_hi)
        for _ in range(24):
            if np.isfinite(y_lo) and np.isfinite(y_hi) and (y_lo * y_hi <= 0):
                break
            k_lo -= 0.5
            k_hi += 0.5
            y_lo, y_hi = f_of_k(k_lo), f_of_k(k_hi)
        else:
            raise RuntimeError("Delta inversion: no bracket found.")

        sol = root_scalar(f_of_k, bracket=[k_lo, k_hi], method="bisect", xtol=tol, maxiter=max_iter)
        if not sol.converged:
            raise RuntimeError("Delta inversion did not converge.")

        K_star = F * np.exp(sol.root)
        return self.iv(K_star, t), K_star