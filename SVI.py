"""
SVI class IV surface builder and evaluator.
Usage overview
- Initialize: `surf = SVI(price_source="price")` (or `"iv"` if you pass IVs)
- Fit: `surf.fit(as_of_t, spot_price, strikes, exp_dates, is_calls, option_values, r=0.04, q=0.0)`
  - `as_of_t`: the time of the option contracts' quotes 'YYYY-mm-dd HH:MM'
  - `spot_price`: spot at `as_of_t`
  - `strikes` (float), `exp_dates` (float), `is_calls` (bool), `option_values` (float): single vals or arrays; all are broadcast to a common length
  - `option_values` are option prices if `price_source="price"`, or implied vols if `price_source="iv"`
- Query IVs: `surf.iv(strikes, T)` where `T` is YEARS to expiry (both float or float array)
- Or query IVs from delta: `surf.iv_from_delta(deltas, T, is_call=None)` (both float or float array)
- both Returns the predicted IV(s) as float or float array matching input shapes
- Use `calculate_T(as_of_t, exp_date)` to compute `T` (time to expiry in years)

example:
surf = SVI(price_source="price")
surf.fit(as_of_t, spot_price, strikes, exp_dates, is_calls, mid_prices, r=0.04, q=0.0)
T = calculate_T(as_of_t, "2025-08-15")
iv_single = surf.iv(350.0, T)
iv_vector = surf.iv(strikes*1.1, T*1.1)

what won't work well:
- u try to use the option contracts' and stock's price CHART data instead of their QUOTE data. 
because chart data is collected only if a transaction happens. the data for the contracts and stock will then be slightly misaligned in time, 
where as the quotes are always there and aligned in time.
- the option quotes themselves have too much spread, and not enough liquidity making for accurate pricing.
- you try to evlauate strikes and times to expiry that are too far away from the bounds of your input data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, List
from scipy import optimize
from scipy.stats import norm

# Black–Scholes hooks (you already have these)
# ---------------------------------------------------------------------
from black_scholes import (
    calculate_option_price as _bs_price,
    calculate_IV as _bs_iv,
    calculate_vega as _bs_vega,
)

def _bs_price_wrap(S, K, T, sigma, r, q, call: bool):
    return _bs_price(S, K, T, sigma, r=r, q=q, call_or_put=("call" if call else "put"))

def _implied_vol_wrap(price, S, K, T, r, q, call: bool):
    return _bs_iv(price, S, K, T, r=r, q=q, call_or_put=("call" if call else "put"))

def _bs_price_vec(S, K, T, sigma, r, q, is_call_bool):
    """Vectorized BS price via two calls to calculate_option_price and selection by call/put mask."""
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call_bool, dtype=bool)
    call_prices = _bs_price(S, K, T, sigma, r=r, q=q, call_or_put="call")
    put_prices = _bs_price(S, K, T, sigma, r=r, q=q, call_or_put="put")
    return np.where(is_call, call_prices, put_prices)

# Utilities
# ---------------------------------------------------------------------
EPS = 1e-12

def forward_from_spot(S: float, T: float, r: float, q: float) -> float:
    return float(S * np.exp((r - q) * T))

def total_var_to_vol(w: float, T: float) -> float:
    return float(np.sqrt(max(w, 0.0) / max(T, EPS)))

def calculate_T(
    as_of_t: str, #YYYY-mm-dd HH:MM
    exp_date: str, #YYYY-mm-dd
    expiry_hour_local: int = 16) -> float:
    t0 = pd.to_datetime(as_of_t)
    t_exp = pd.to_datetime(exp_date) + pd.Timedelta(hours=expiry_hour_local)
    return float(max((t_exp - t0).total_seconds() / (365.0 * 24.0 * 3600.0), 1e-6))

# Minimal Raw-SVI slice
# ---------------------------------------------------------------------
@dataclass
class SVIRaw:
    a: float
    b: float
    rho: float      # |rho| < 1
    m: float
    sigma: float    # > 0

    def w(self, k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        k = np.asarray(k)
        out = self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2))
        return out if out.ndim else float(out)

# Quote container
# ---------------------------------------------------------------------
@dataclass
class Quote:
    K: float
    value: float     # price or IV depending on is_iv
    is_call: bool
    is_iv: bool

# Single-slice calibration (vega-weighted price fit; robust seed)
# --------------------------------------------------------------------
def calibrate_raw_slice2(
    S: float, T: float, r: float, q: float,
    quotes: Sequence[Quote],
    bounds: Tuple[Tuple[float, float, float, float, float], Tuple[float, float, float, float, float]] = (
        (-5.0, 1e-12, -0.999, -5.0, 1e-5),
        ( 5.0, 10.0,   0.999,  5.0,  5.0)
    ),
) -> SVIRaw:

    F = forward_from_spot(S, T, r, q)

    # Arrays per quote
    Ks_arr   = np.array([qte.K for qte in quotes], dtype=float)
    ks       = np.log(Ks_arr / F)  # log-moneyness
    is_call  = np.array([qte.is_call for qte in quotes], dtype=bool)

    # Convert inputs to PRICES once
    prices = np.array([
    _bs_price_wrap(S, qt.K, T, qt.value, r, q, qt.is_call) if qt.is_iv else qt.value
    for qt in quotes
    ], dtype=float)

    iv0 = np.empty(len(quotes), dtype=float)
    for i, qte in enumerate(quotes):
        if qte.is_iv:
            iv0[i] = float(qte.value)
        else:
            try:
                iv0[i] = _implied_vol_wrap(prices[i], S, qte.K, T, r, q, qte.is_call)
                if not np.isfinite(iv0[i]) or iv0[i] <= 0: iv0[i] = 0.2
            except Exception:
                iv0[i] = 0.2

    # Vega weights (cap away from 0)
    vegas_w = np.maximum(_bs_vega(S, Ks_arr, T, iv0, r=r, q=q), 1e-6)

    # Seed from IVs (nearest ATM, small-window slope)
    if np.sum(np.isfinite(iv0) & (iv0 > 0)) >= 2:
        idx_atm = int(np.argmin(np.abs(ks)))
        sigma_atm = float(iv0[idx_atm])
        w0 = sigma_atm**2 * T

        mask = (np.abs(ks) <= 0.2)
        if mask.sum() < 2: mask = slice(None)
        slope, _ = np.polyfit(ks[mask], iv0[mask], 1)
        wprime0 = 2.0 * np.sqrt(max(w0 * T, EPS)) * float(slope)

        rho0 = -0.3 if wprime0 <= 0 else 0.3
        b0 = float(np.clip(abs(wprime0) / max(abs(rho0), 1e-6), 1e-4, 10.0))
        m0 = 0.0
        sigma_param0 = 0.3
        a0 = w0 - b0 * sigma_param0

        min_w = a0 + b0 * sigma_param0 * np.sqrt(max(1.0 - rho0**2, 0.0))
        if min_w < 1e-6:
            a0 += (1e-6 - min_w) + 1e-6
        x0 = np.array([a0, b0, rho0, m0, sigma_param0], dtype=float)
    else:
        sigma_atm = 0.25
        w0 = sigma_atm**2 * T
        x0 = np.array([w0 - 0.05*0.3, 0.05, -0.3, 0.0, 0.3], dtype=float)

    lo, hi = [np.array(b) for b in bounds]
    x0 = np.minimum(np.maximum(x0, lo + 1e-12), hi - 1e-12)

    # Vector helpers
    def _raw_from_x(x):
        return SVIRaw(a=float(x[0]), b=float(x[1]), rho=float(x[2]), m=float(x[3]), sigma=float(x[4]))

    def _w_of_k(x, k):  # vectorized w(k)
        a, b, rho, m, s = x
        X = k - m
        return a + b*(rho*X + np.sqrt(X*X + s*s))

    def _model_prices(x):
        w = _w_of_k(x, ks)
        sig = np.sqrt(np.maximum(w, 1e-18) / T)
        return _bs_price_vec(S, Ks_arr, T, sig, r, q, is_call)

    # Residuals (vectorized)
    def objective(x):
        # quick reject invalid regions (keeps TRF happy)
        if not (-0.999 < x[2] < 0.999) or x[1] <= 0 or x[4] <= 0:
            return 1e8 * np.ones(len(quotes) + 1)

        # price residuals
        C = _model_prices(x)
        res = (C - prices) / vegas_w

        # one penalty residual to keep min total variance >= 0
        a, b, rho, m, s = x
        min_w = a + b*s*np.sqrt(max(1.0 - rho*rho, 0.0))
        pen = max(0.0, -min_w)
        return np.append(res, 1e3 * pen*pen)

    # Analytic Jacobian
    def jacobian(x):
        a, b, rho, m, s = x
        X = ks - m
        sqrt_term = np.sqrt(X*X + s*s)

        # dw/dθ pieces
        dw_da = np.ones_like(ks)
        dw_db = rho*X + sqrt_term
        dw_dr = b * X
        dw_dm = b * (-rho - X / np.maximum(sqrt_term, 1e-16))
        dw_ds = b * (s / np.maximum(sqrt_term, 1e-16))

        w = a + b*(rho*X + sqrt_term)
        sig = np.sqrt(np.maximum(w, 1e-18) / T)

        # dC/dw = vega / (2 σ T)
        vega_model = _bs_vega(S, Ks_arr, T, sig, r=r, q=q)
        dC_dw = vega_model / (2.0 * np.maximum(sig, 1e-16) * T)

        scale = dC_dw / vegas_w  # elementwise

        J = np.empty((len(ks) + 1, 5), dtype=float)
        J[:-1, 0] = scale * dw_da
        J[:-1, 1] = scale * dw_db
        J[:-1, 2] = scale * dw_dr
        J[:-1, 3] = scale * dw_dm
        J[:-1, 4] = scale * dw_ds

        # Penalty row
        min_w = a + b*s*np.sqrt(max(1.0 - rho*rho, 0.0))
        if min_w >= 0:
            J[-1, :] = 0.0
        else:
            sqrt_ = np.sqrt(max(1.0 - rho*rho, 1e-16))
            dmin = np.array([
                1.0,               # d/da
                s*sqrt_,           # d/db
                b*s*(-rho / max(sqrt_, 1e-16)),  # d/drho
                0.0,               # d/dm
                b*sqrt_,           # d/dsigma
            ], dtype=float)
            pen = -min_w
            # J[-1, :] = 2e3 * pen * dmin
            J[-1, :] = -2e3 * pen * dmin
        return J

    sol = optimize.least_squares(
        objective, x0, jac=jacobian, bounds=(lo, hi), method="trf",
        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=4000
    )
    return _raw_from_x(sol.x)

# MAIN CLASS — minimal surface builder & evaluator
# ---------------------------------------------------------------------
class SVI:
    """
    Minimal IV surface:
      - __init__(price_source='price' or 'iv')
      - .fit(as_of_t, spot_price, strikes, exp_dates, is_calls, values, r=0.04, q=0.0)
      - .iv(strikes, T) -> implied vol(s)
    """

    def __init__(self, price_source: str = "price"):
        assert price_source in ("price", "iv")
        self.price_source = price_source

        self.S: Optional[float] = None #spot price at as_of_t
        self.as_of_t: Optional[str] = None #YYYY-mm-dd HH:MM
        self.r: Optional[float] = 0.04 #risk-free rate
        self.q: Optional[float] = 0.0 #dividend yield

        # calibrated slices: list of (T, SVIRaw, exp_date)
        self.slices: List[Tuple[float, SVIRaw, str]] = []
        self.theta = None  # ATM total variance curve

    # --- internal: broadcast and validate inputs ---
    @staticmethod
    def _as_array(x, N: Optional[int] = None, dtype=None):
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            arr = np.asarray(x if not isinstance(x, pd.Series) else x.values, dtype=dtype)
        else:
            # scalar
            if N is None:
                arr = np.array([x], dtype=dtype)
            else:
                arr = np.full(shape=(N,), fill_value=x, dtype=dtype)
        return arr

    @staticmethod
    def _k_from_K(F: float, K: float) -> float:
        return float(np.log(K / F))

    def _theta_builder(self):
        """
        Build θ(t) from calibrated slices: θ_j = w_j(0).
        Interpolate in sqrt(θ). For t < T1 or t > Tn, extrapolate with constant ATM variance.
        """
        Ts = np.array([T for (T, _, _) in self.slices], dtype=float)
        thetas = np.array([sl.w(0.0) for (_, sl, _) in self.slices], dtype=float)
        thetas = np.maximum.accumulate(thetas)  # enforce monotone
        sqrt_th = np.sqrt(np.maximum(thetas, 0.0))

        T1, Tn = Ts[0], Ts[-1]
        v1 = thetas[0] / max(T1, EPS)
        vn = thetas[-1] / max(Tn, EPS)

        def θ(t: float) -> float:
            t = float(t)
            if t <= T1:  # go through origin with variance v1
                return float(max(v1 * t, 0.0))
            if t >= Tn:  # continue with last ATM variance
                return float(max(vn * t, 0.0))
            j = np.searchsorted(Ts, t)
            Tl, Tr = Ts[j-1], Ts[j]
            xl, xr = sqrt_th[j-1], sqrt_th[j]
            w = (t - Tl) / max(Tr - Tl, EPS)
            return float(((1 - w) * xl + w * xr)**2)
        return θ

    def _interpolate_price_at(self, k: float, t: float) -> float:
        Ts = [T for (T, _, _) in self.slices]
        t = float(t)
        
        # --- Extrapolation: t is AFTER the last calibrated slice ---
        # Paper (Sec 5.3): w(k, θ_t) = w(k, θ_tn) + θ_t - θ_tn
        if t >= Ts[-1]:
            Tn, sln, _ = self.slices[-1]
            if abs(t - Tn) < 1e-9: # Handle near-exact match
                t = Tn

            w_n_k = sln.w(k)
            theta_n = self.theta(Tn)
            theta_t = self.theta(t)
            
            # This is the arbitrage-free extrapolation formula from Theorem 4.3
            w_t_k = w_n_k + (theta_t - theta_n)
            
            sigma = total_var_to_vol(w_t_k, t)
            Ft = forward_from_spot(self.S, t, self.r, self.q)
            Kt = Ft * np.exp(k)
            return _bs_price_wrap(self.S, Kt, t, sigma, self.r, self.q, call=True)

        # --- Extrapolation: t is BEFORE the first calibrated slice ---
        # Paper (Sec 5.3): Interpolate between t=0 (intrinsic value) and t=t1
        if t <= Ts[0]:
            # --- Extrapolation: t is BEFORE the first calibrated slice ---
            T1, sl1, _ = self.slices[0]
            if abs(t - T1) < 1e-9:
                t = T1

            # Fixed-k strikes at each time
            F0 = self.S                      # F_0
            K0 = F0 * np.exp(k)
            F1 = forward_from_spot(self.S, T1, self.r, self.q)
            K1 = F1 * np.exp(k)
            Ft = forward_from_spot(self.S, t,  self.r, self.q)
            Kt = Ft * np.exp(k)

            # Undiscounted endpoints:
            # t = 0 intrinsic is already undiscounted
            C0_undisc = max(self.S - K0, 0.0)

            # First-slice price from SVI, then lift to undiscounted
            sigma1 = total_var_to_vol(sl1.w(k), T1)
            C1_pv = _bs_price_wrap(self.S, K1, T1, sigma1, self.r, self.q, call=True)
            C1_undisc = C1_pv * np.exp(self.r * T1)

            # Square-root weight
            theta_t = self.theta(t)
            theta1  = self.theta(T1)
            sqrt_theta1 = np.sqrt(max(theta1, 0.0))
            alpha = (sqrt_theta1 - np.sqrt(max(theta_t, 0.0))) / max(sqrt_theta1, EPS)

            # Interpolate in UNDIsCOUNTED C/K, then discount back
            C_over_K_undisc_t = alpha * (C0_undisc / max(K0, EPS)) + (1.0 - alpha) * (C1_undisc / max(K1, EPS))
            C_pv_t = np.exp(-self.r * t) * C_over_K_undisc_t * Kt
            return float(C_pv_t)

        # --- Interpolation: t is BETWEEN two calibrated slices ---
        # This part was already correct and remains the same.
        idx_right = np.searchsorted(Ts, t)
        T1, sl1, _ = self.slices[idx_right - 1]
        T2, sl2, _ = self.slices[idx_right]

        # Handle exact match on an interior slice
        if abs(t - T1) <= 1e-12:
            sigma = total_var_to_vol(sl1.w(k), T1)
            Ft = forward_from_spot(self.S, T1, self.r, self.q)
            Kt = Ft * np.exp(k)
            return _bs_price_wrap(self.S, Kt, T1, sigma, self.r, self.q, call=True)
        
        θt, θ1, θ2 = self.theta(t), self.theta(T1), self.theta(T2)
        sqrt_diff = np.sqrt(max(θ2,0.0)) - np.sqrt(max(θ1,0.0))
        α = (np.sqrt(max(θ2,0.0)) - np.sqrt(max(θt,0.0))) / max(sqrt_diff, EPS)

        F1 = forward_from_spot(self.S, T1, self.r, self.q); K1 = F1 * np.exp(k)
        F2 = forward_from_spot(self.S, T2, self.r, self.q); K2 = F2 * np.exp(k)
        Ft = forward_from_spot(self.S, t,  self.r, self.q); Kt = Ft * np.exp(k)

        σ1 = total_var_to_vol(sl1.w(k), T1)
        σ2 = total_var_to_vol(sl2.w(k), T2)
        C1_pv = _bs_price_wrap(self.S, K1, T1, σ1, self.r, self.q, call=True)
        C2_pv = _bs_price_wrap(self.S, K2, T2, σ2, self.r, self.q, call=True)
        C1_undisc = C1_pv * np.exp(self.r * T1)
        C2_undisc = C2_pv * np.exp(self.r * T2)

        C_over_K_undisc_t = α * (C1_undisc / K1) + (1.0 - α) * (C2_undisc / K2)
        C_pv_t = np.exp(-self.r * t) * C_over_K_undisc_t * Kt
        return float(C_pv_t)

    # ------------------ PUBLIC API ------------------

    def fit(self, as_of_t: str,
        spot_price: float,
        strikes: Sequence[float],
        exp_dates: Union[str, Sequence[str]],
        is_calls: Union[bool, Sequence[bool]],
        option_values: Union[float, Sequence[float]],
        r: float = 0.04,
        q: float = 0.0
        ) -> "SVI":

        """       
        Fit one Raw-SVI slice per expiry from raw observations you pass in.
        note, for the option_values and spot_price, its crucial that they are aligned in time. 
        best to use the bid / ask / mid quotes at as_of_t for both. i use polygon.io API for this.

        Parameters
        ----------
        as_of_t : str
            time of the quotes of the option contracts 'YYYY-mm-dd HH:MM'.
        spot_price : float
            Spot price at as_of_t.
        strikes : array[float]
            Strike(s) for each observation.
        exp_dates : str or array[str]
            Expiry date(s) 'YYYY-mm-dd' for each observation.
        is_calls : bool or array[bool]
            True for call, False for put – per observation (or a scalar to apply to all).
        option_values : float or array
            Observed values per observation – interpreted by price_source:
              - 'price' -> option prices
              - 'iv'        -> implied volatilities
        r, q : floats
            Risk-free rate and dividend yield (annualized, cont.).
        """
        # Store globals
        self.S = float(spot_price)
        self.as_of_t = str(as_of_t)
        self.r = float(r)
        self.q = float(q)

        # Broadcast all inputs to common length N
        # Determine N from the longest provided sequence among strikes/exp_dates/is_calls/values
        def _len(x):
            if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
                return len(x)
            return 1

        N = max(_len(strikes), _len(exp_dates), _len(is_calls), _len(option_values))
        Ks   = self._as_array(strikes, N=N, dtype=float)
        EDs  = self._as_array(exp_dates, N=N, dtype=object).astype(str)
        CPs  = self._as_array(is_calls, N=N, dtype=bool)
        Vals = self._as_array(option_values, N=N, dtype=float)

        # Validate lengths
        for name, arr in (("strikes", Ks), ("exp_dates", EDs), ("is_calls", CPs), ("values", Vals)):
            if len(arr) != N:
                raise ValueError(f"{name} length mismatch after broadcasting.")

        # Group by expiry and calibrate a slice per expiry using ALL rows for that expiry
        self.slices = []
        unique_exp = sorted(pd.unique(EDs), key=lambda d: pd.to_datetime(d))

        for exp in unique_exp:
            mask = (EDs == exp)
            if not np.any(mask):
                continue
            T = calculate_T(self.as_of_t, exp)
            use_iv = (self.price_source == "iv")
            quotes: List[Quote] = [
                Quote(K=float(K), value=float(val), is_call=bool(cp), is_iv=use_iv)
                for K, val, cp in zip(Ks[mask], Vals[mask], CPs[mask])
            ]
            sl = calibrate_raw_slice2(S=self.S, T=T, r=self.r, q=self.q, quotes=quotes)
            self.slices.append((T, sl, exp))

        if not self.slices:
            raise RuntimeError("No slices calibrated; check your inputs.")

        # Sort by T and build θ(t)
        self.slices.sort(key=lambda tup: tup[0])
        self.theta = self._theta_builder()
        return self



    '''------the follwing functions are for helper funcs for the iv from delta func-------'''
    def _sigma_and_K_at_k(self, k: float, t: float) -> Tuple[float, float]:
        """
        Given log-moneyness k and time t, return (sigma, K) where sigma is the
        Black-Scholes IV implied by your SVI surface (via price interpolation).
        """
        price = self._interpolate_price_at(k=k, t=t)  # call PV at (k,t)
        Ft = forward_from_spot(self.S, t, self.r, self.q)
        K = float(Ft * np.exp(k))
        sig = _implied_vol_wrap(price, self.S, K, t, self.r, self.q, call=True)
        # Numerical guard
        if not np.isfinite(sig) or sig <= 0.0:
            sig = 1e-8
        return sig, K

    def _delta_at_k(self, k: float, t: float, is_call: bool) -> float:
        """
        Compute Black-Scholes delta at (k,t) using the SVI-implied sigma(k,t).
        """
        sig, K = self._sigma_and_K_at_k(k, t)
        sqrtT = np.sqrt(max(t, EPS))
        # Forward delta: N(d1_fwd) for calls, N(d1_fwd)-1 for puts,
        # with d1_fwd = (ln(F/K) + 0.5*sig^2*T) / (sig*sqrt(T)) = (-k + 0.5*w)/sqrt(w).
        d1f = (-k + 0.5 * sig * sig * t) / max(sig * sqrtT, 1e-16)
        return norm.cdf(d1f) if is_call else (norm.cdf(d1f) - 1.0)
    def _solve_k_for_delta(
        self,
        delta_target: float,
        t: float,
        is_call: Optional[bool],
        bracket: Tuple[float, float] = (-5.0, 5.0),
        tol: float = 1e-10,
        max_expand: int = 7,
    ) -> Tuple[float, float, float]:
        """
        Solve for k such that BS delta(k,t) matches delta_target under the chosen convention.
        Returns (k*, sigma(k*,t), K(k*)).
        """
        if self.S is None or not self.slices or self.theta is None:
            raise RuntimeError("call fit before iv_from_delta")

        if is_call is None:
            is_call = (delta_target >= 0.0)

        if is_call:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = -1.0, 0.0
        eps = 1e-10
        delta_tgt = float(np.clip(delta_target, lo + eps, hi - eps))

        def g(k: float) -> float:
            return self._delta_at_k(k, t, is_call) - delta_tgt

        # Try to bracket a root
        klo, khi = bracket
        glo, ghi = g(klo), g(khi)
        expand = 0
        while not (np.isfinite(glo) and np.isfinite(ghi) and glo * ghi <= 0.0) and expand < max_expand:
            klo -= 2.0
            khi += 2.0
            glo, ghi = g(klo), g(khi)
            expand += 1

        # If still not bracketed, do a coarse grid search for a sign change
        if not (np.isfinite(glo) and np.isfinite(ghi) and glo * ghi <= 0.0):
            ks = np.linspace(-12.0, 12.0, 97)
            gs = np.array([g(x) for x in ks])
            # Find any adjacent sign change
            sign_changes = np.where(np.isfinite(gs[:-1] * gs[1:]) & (gs[:-1] * gs[1:] <= 0.0))[0]
            if sign_changes.size > 0:
                i = int(sign_changes[0])
                klo, khi = ks[i], ks[i + 1]
            else:
                # No sign change found; pick the best k by minimizing |g|
                i_best = int(np.nanargmin(np.abs(gs)))
                k_star = float(ks[i_best])
                sig_star, K_star = self._sigma_and_K_at_k(k_star, t)
                return k_star, sig_star, K_star

        # Root find (Brent is robust for monotone g)
        k_star = float(optimize.brentq(g, klo, khi, xtol=tol, rtol=tol, maxiter=200))
        sig_star, K_star = self._sigma_and_K_at_k(k_star, t)
        return k_star, sig_star, K_star
    def strike_from_delta(
        self,
        deltas: Union[float, Sequence[float]],
        T:       Union[float, Sequence[float]],
        is_call: Optional[Union[bool, Sequence[bool]]] = None,
    ) -> Union[float, np.ndarray]:
        """
        Return strike(s) corresponding to target delta(s) at maturity T. (float or np.ndarray)
        """
        Ds = np.atleast_1d(np.asarray(deltas, dtype=float))
        Ts = np.atleast_1d(np.asarray(T, dtype=float))
        if is_call is None:
            Cs = np.array([d >= 0.0 for d in Ds], dtype=bool)
        else:
            Cs = np.atleast_1d(np.asarray(is_call, dtype=bool))
        Ds, Ts, Cs = np.broadcast_arrays(Ds, Ts, Cs)

        out = np.empty_like(Ds, dtype=float)
        for idx, d in np.ndenumerate(Ds):
            t = float(Ts[idx])
            c = bool(Cs[idx])
            k_star, _, K_star = self._solve_k_for_delta(float(d), t, c)
            out[idx] = K_star
        return float(out.item()) if out.size == 1 else out
    '''---------------------------------------------------------------------------'''
    

    def iv_from_delta(
        self,
        deltas: Union[float, Sequence[float]],
        T:       Union[float, Sequence[float]],
        is_call: Optional[Union[bool, Sequence[bool]]] = None,
        return_strike: bool = False,
    ) -> Union[float, Tuple[float, float], np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Return IV(s) that correspond to a target delta at maturity T, by inverting for k.
        Returns IVs or (IVs, Ks), Shape matches the broadcast of (deltas, T[, is_call]).
        """
        Ds = np.atleast_1d(np.asarray(deltas, dtype=float))
        Ts = np.atleast_1d(np.asarray(T, dtype=float))
        if is_call is None:
            Cs = np.array([d >= 0.0 for d in Ds], dtype=bool)
        else:
            Cs = np.atleast_1d(np.asarray(is_call, dtype=bool))
        Ds, Ts, Cs = np.broadcast_arrays(Ds, Ts, Cs)

        ivs = np.empty_like(Ds, dtype=float)
        Ks  = np.empty_like(Ds, dtype=float) if return_strike else None

        for idx, d in np.ndenumerate(Ds):
            t = float(Ts[idx])
            c = bool(Cs[idx])
            k_star, sig_star, K_star = self._solve_k_for_delta(float(d), t, c)
            ivs[idx] = sig_star
            if return_strike:
                Ks[idx] = K_star

        if ivs.size == 1:
            return (float(ivs.item()), float(Ks.item())) if return_strike else float(ivs.item())
        else:
            return (ivs, Ks) if return_strike else ivs

    def iv(
        self,
        strikes: Union[float, Sequence[float]],
        T:       Union[float, Sequence[float]],
    ) -> Union[float, np.ndarray]:
        """
        Return implied vol(s) at (strike, T). Supports scalars or arrays (broadcasted).
        """
        if self.S is None or not self.slices or self.theta is None:
            raise RuntimeError("Call fit(...) before iv(...).")

        Ks = np.atleast_1d(np.asarray(strikes, dtype=float))
        Ts = np.atleast_1d(np.asarray(T, dtype=float))
        Ks, Ts = np.broadcast_arrays(Ks, Ts)

        out = np.empty_like(Ks, dtype=float)
        for idx, K_i in np.ndenumerate(Ks):
            t_i = float(Ts[idx])
            Ft = forward_from_spot(self.S, t_i, self.r, self.q)
            k = np.log(K_i / Ft)
            # Interpolate to a call price at (k, t_i), then invert to IV
            price_t = self._interpolate_price_at(k=k, t=t_i)
            out[idx] = _implied_vol_wrap(price_t, self.S, K_i, t_i, self.r, self.q, call=True)

        return float(out.item()) if out.size == 1 else out


if __name__=="__main__":
    '''EXAMPLE:'''

    import pandas as pd
    import sys,os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from polygon_data.downloading_option_chain import fetch_option_chain


    close_t = "2025-08-15 15:55"
    ticker  = "TSLA"
    strike_from, strike_to = 280, 375
    exp_date = "2025-08-15"
    r,q=0.04,0.00
    call_or_put='call'
    extra_weeks=2

    #fetching the historical snapshot of the quotes of an option chain
    tbl=fetch_option_chain(
        close_t=close_t, ticker=ticker, exp_date=exp_date,
        strike_from=strike_from, strike_to=strike_to,
        call_or_put=call_or_put,
        extra_weeks=extra_weeks,
        r=r, q=q, price_source="mid_price"
    )
    strikes=tbl['strike'].to_numpy()
    exp_dates=tbl['exp_date'].to_numpy()
    prices=tbl['mid_price'].to_numpy()
    spot_price=tbl['stock_mid_price'].to_numpy()[0]
    is_calls=np.ones(len(strikes),dtype=bool)
    IVs=tbl['iv'].to_numpy()

    #create the IV surface with the data above
    surf=SVI(price_source="price")
    surf.fit(close_t,spot_price,strikes,exp_dates,is_calls,prices,r=r,q=q)

    #see predicted IVs and prices for the same contracts as above
    T_s=np.array([calculate_T(close_t,exp) for exp in exp_dates])
    pred_ivs=surf.iv(strikes,T_s)    
    pred_prices=_bs_price(spot_price,strikes,T_s,pred_ivs,r=r,q=q,call_or_put=call_or_put)
    price_errors=np.abs(pred_prices-prices)
    iv_errors=np.abs(pred_ivs-IVs)

    # Create a new DataFrame to compare predicted and actual data
    comparison_df = pd.DataFrame({
        'Strike': strikes,
        'Expiration Date': exp_dates,
        'Predicted IV': pred_ivs,
        'Actual IV': IVs,
        'IV Error': iv_errors,
        'Predicted Price': pred_prices,
        'Actual Price': prices,
        'Price Error': price_errors
    })
    print(comparison_df.to_string(index=False)) 








