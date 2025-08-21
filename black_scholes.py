"""
Black‑Scholes utilities

Common parameters (floats or array‑like; broadcasting supported where noted):
- S: Spot price
- K: Strike price
- T: Time to expiry (in years)
- IV: Implied Volatility
- r: risk‑free rate
- q: dividend yield

Vectorization:
- The following functions accept either scalars or array‑like inputs (any mix, via NumPy
  broadcasting). They return a Python scalar if all numeric inputs are scalars; otherwise
  they return an ndarray:
  - calculate_option_price
  - calculate_delta
  - calculate_vega
  - calculate_theta
  - calculate_gamma

- Functions that use root‑finding (brentq/newton) are scalar‑only:
  - calculate_IV
  - calculate_stock_price_from_option
  - calculate_strike_for_delta
"""
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from numpy import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq, newton

_EPS = 1e-10  # numeric floor
_IV_MIN, _IV_MAX = 1e-6, 10.0
_IV_MAX_HARD = 50.0

def _is_scalar_like(x):
    return np.isscalar(x) or (isinstance(x, np.ndarray) and x.shape == ())

def _maybe_scalar(inputs, out):
    """Return a Python scalar if all numeric inputs are scalar‑like; else return as is."""
    if all(_is_scalar_like(x) for x in inputs):
        return np.asarray(out).item() if isinstance(out, np.ndarray) else out
    return out

def _d1_d2(S, K, T, IV, r, q):
    """
    Return (d1, d2) for Black‑Scholes.
    Returns:  float or ndarray (d1, d2)
    """
    T = np.maximum(T, _EPS)
    IV = np.maximum(IV, _EPS)
    sig_sqrtT = IV * sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * IV ** 2) * T) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    return d1, d2

def calculate_option_price(S, K, T, IV, r=0.04, q=0.0, call_or_put="call"):
    """Return Black‑Scholes option price (call or put).
    Accepts scalars or array‑like inputs (broadcasted). Returns scalar if all inputs are scalars.
    """
    call_or_put = call_or_put.lower()
    if call_or_put not in {"call", "put"}:
        raise ValueError("opt_type must be 'call' or 'put'")

    T_arr = np.asarray(T)
    d1, d2 = _d1_d2(S, K, T_arr, IV, r, q)
    df_r, df_q = exp(-r * T_arr), exp(-q * T_arr)

    if call_or_put == "call":
        price = df_q * S * norm.cdf(d1) - df_r * K * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:  # put
        price = df_r * K * norm.cdf(-d2) - df_q * S * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)

    out = np.where(T_arr <= 0, intrinsic, price)
    return _maybe_scalar((S, K, T, IV, r, q), out)

def calculate_IV(option_price, S, K, T, r=0.04, q=0.0, call_or_put="call",
                 xtol=1e-4, max_iter=100):
    """Returns: float (implied volatility)"""
    call_or_put = call_or_put.lower()

    # --- Arbitrage bound check ------------------------------------------------
    df_r, df_q = exp(-r * T), exp(-q * T)
    if call_or_put == "call":
        lower, upper = max(0.0, S*df_q - K*df_r), S*df_q
    else:  # put
        lower, upper = max(0.0, K*df_r - S*df_q), K*df_r
    if not (lower - 1e-12 <= option_price <= upper + 1e-12):
        return np.nan  # or raise ValueError

    # --- Root‑finding target --------------------------------------------------
    def diff(sigma):
        return calculate_option_price(S, K, T, sigma, r, q, call_or_put) - option_price

    # --- Expand the upper bracket if needed ----------------------------------
    a, b = _IV_MIN, _IV_MAX
    fa, fb = diff(a), diff(b)
    while fa * fb > 0 and b < _IV_MAX_HARD:
        b *= 2
        fb = diff(b)

    if fa * fb > 0:
        return np.nan  # bracket never found → probably bad input

    # --- Brent (robust) -------------------------------------------------------
    try:
        sigma_star = brentq(diff, a, b, xtol=xtol, maxiter=max_iter//2)
    except ValueError:          # extremely flat diff curve
        return np.nan
    # --- One Newton step for polish ------------------------------------------
    try:
        sigma_star = newton(diff, sigma_star,
                            fprime=lambda s: calculate_vega(S, K, T, s, r, q),
                            tol=xtol, maxiter=max_iter//2)
    except RuntimeError:
        pass  # keep Brent result if Newton fails

    return sigma_star

def calculate_stock_price_from_option(option_price, K, T, IV, r=0.04, q=0.0,
                                      option_type="call", tol=1e-2, max_iter=30,
                                      method="brent"):
    """Invert BS for S.  `method="brent"` (robust) or `"newton"` (fast)."""

    def f(S):
        return calculate_option_price(S, K, T, IV, r, q, option_type) - option_price

    if method == "newton":
        # Use Δ as derivative; a single Newton step is usually enough
        def fprime(S):
            return calculate_delta(S, K, T, IV, r, q, option_type)
        guess = K * 1.0  # ATM starting point
        return newton(f, guess, fprime=fprime, tol=tol, maxiter=max_iter)
    else:
        # BrentQ is ~5-10× faster than pure-Python bisection
        bracket = (1e-3, K * 10.0) if option_type == "call" else (1e-3, K * 10.0)
        return brentq(f, *bracket, xtol=tol, maxiter=max_iter)



# Greeks
# -------------------------------------------------------------------

def calculate_vega(S, K, T, IV, r=0.04, q=0.0):
    """Return Black-Scholes Vega (per 1.0 change in volatility).
    Returns: float or ndarray
    """
    T_arr = np.asarray(T)
    d1, _ = _d1_d2(S, K, T_arr, IV, r, q)
    vega = S * exp(-q * T_arr) * norm.pdf(d1) * sqrt(T_arr)
    out = np.where(T_arr <= 0, 0.0, vega)
    return _maybe_scalar((S, K, T, IV, r, q), out)

def calculate_delta(S, K, T, IV, r=0.04, q=0.0, option_type="call"):
    T_arr = np.asarray(T)
    d1, _ = _d1_d2(S, K, T_arr, IV, r, q)
    base = exp(-q * T_arr) * norm.cdf(d1)
    out = base if option_type == "call" else base - exp(-q * T_arr)
    return _maybe_scalar((S, K, T, IV, r, q), out)

def calculate_theta(S, K, T, IV, r=0.04, q=0.0, option_type="call"):
    """Return Black-Scholes Theta (per year).
    Accepts scalars or array‑like inputs. Returns scalar if all inputs are scalars.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    T_arr = np.asarray(T)
    T_eff = np.maximum(T_arr, _EPS)
    d1, d2 = _d1_d2(S, K, T_eff, IV, r, q)
    df_r, df_q = exp(-r * T_eff), exp(-q * T_eff)
    first = -(S * df_q * norm.pdf(d1) * IV) / (2.0 * sqrt(T_eff))

    if option_type == "call":
        second = -r * K * df_r * norm.cdf(d2)
        third = q * S * df_q * norm.cdf(d1)
        theta = first + second + third
    else:
        second = r * K * df_r * norm.cdf(-d2)
        third = -q * S * df_q * norm.cdf(-d1)
        theta = first + second + third

    out = np.where(T_arr <= 0, 0.0, theta)
    return _maybe_scalar((S, K, T, IV, r, q), out)

def calculate_gamma(S, K, T, IV, r=0.04, q=0.0):
    """Return Black-Scholes Gamma (same for calls and puts).
    Accepts scalars or array‑like inputs. Returns scalar if all inputs are scalars.
    """
    T_arr = np.asarray(T)
    T_eff = np.maximum(T_arr, _EPS)
    d1, _ = _d1_d2(S, K, T_eff, IV, r, q)
    gamma = exp(-q * T_eff) * norm.pdf(d1) / (S * IV * sqrt(T_eff))
    out = np.where(T_arr <= 0, 0.0, gamma)
    return _maybe_scalar((S, K, T, IV, r, q), out)


#backward solve for strike from greeks
# -------------------------------------------------------------------
def calculate_strike_for_delta(delta_target, S, T, IV, r=0.04, q=0.0,
                               option_type="call", tol=1e-2, max_iter=30):
    """Solve Δ(K)=target using Brent; works for calls *and* puts."""

    # Build a bracket that always captures the root
    # For calls Δ→1 as K→0; for puts Δ→-1 as K→∞
    low, high = 1e-6, S * 10.0
    def f(K):
        return calculate_delta(S, K, T, IV, r, q, option_type) - delta_target

    # Expand bracket until sign change
    f_low, f_high = f(low), f(high)
    while f_low * f_high > 0:
        if option_type == "call":
            high *= 2.0
        else:
            low /= 2.0
        f_low, f_high = f(low), f(high)

    return brentq(f, low, high, xtol=tol, maxiter=max_iter)
