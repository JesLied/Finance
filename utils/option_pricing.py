from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm


EPSILON = 1e-12
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass
class SurfaceGrid:
    ttm_grid: np.ndarray
    log_moneyness_grid: np.ndarray
    iv_grid: np.ndarray
    total_variance_grid: np.ndarray
    local_vol_grid: np.ndarray | None = None


def _as_array(value):
    return np.asarray(value, dtype=float)


def _broadcast_option_type(option_type, shape):
    if np.isscalar(option_type):
        return np.full(shape, str(option_type).upper(), dtype=object)
    return np.asarray(option_type, dtype=object).astype(str)


def _sanitize_inputs(spot, strike, ttm, vol):
    spot = np.maximum(_as_array(spot), EPSILON)
    strike = np.maximum(_as_array(strike), EPSILON)
    ttm = np.maximum(_as_array(ttm), EPSILON)
    vol = np.maximum(_as_array(vol), EPSILON)
    return np.broadcast_arrays(spot, strike, ttm, vol)


def black_scholes_d1_d2(spot, strike, ttm, rate, vol):
    spot, strike, ttm, vol = _sanitize_inputs(spot, strike, ttm, vol)
    rate = np.broadcast_to(_as_array(rate), spot.shape)
    sqrt_t = np.sqrt(ttm)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def black_scholes_price(spot, strike, ttm, rate, vol, option_type):
    d1, d2 = black_scholes_d1_d2(spot, strike, ttm, rate, vol)
    spot, strike, ttm, vol = _sanitize_inputs(spot, strike, ttm, vol)
    rate = np.broadcast_to(_as_array(rate), spot.shape)
    discount = np.exp(-rate * ttm)
    call = spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    put = strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)
    option_type = _broadcast_option_type(option_type, call.shape)
    return np.where(option_type == "CALL", call, put)


def black_scholes_greeks(spot, strike, ttm, rate, vol, option_type):
    d1, d2 = black_scholes_d1_d2(spot, strike, ttm, rate, vol)
    spot, strike, ttm, vol = _sanitize_inputs(spot, strike, ttm, vol)
    rate = np.broadcast_to(_as_array(rate), spot.shape)
    option_type = _broadcast_option_type(option_type, spot.shape)
    sqrt_t = np.sqrt(ttm)
    discount = np.exp(-rate * ttm)
    pdf_d1 = norm.pdf(d1)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    theta_call = (
        -(spot * pdf_d1 * vol) / (2.0 * sqrt_t)
        - rate * strike * discount * norm.cdf(d2)
    )
    theta_put = (
        -(spot * pdf_d1 * vol) / (2.0 * sqrt_t)
        + rate * strike * discount * norm.cdf(-d2)
    )
    rho_call = strike * ttm * discount * norm.cdf(d2)
    rho_put = -strike * ttm * discount * norm.cdf(-d2)

    delta = np.where(option_type == "CALL", delta_call, delta_put)
    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t
    theta = np.where(option_type == "CALL", theta_call, theta_put)
    rho = np.where(option_type == "CALL", rho_call, rho_put)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }


def implied_volatility(price, spot, strike, ttm, rate, option_type, initial_vol=0.5, tol=1e-8, max_iter=100):
    price = np.maximum(_as_array(price), 0.0)
    spot, strike, ttm, _ = _sanitize_inputs(spot, strike, ttm, np.asarray(initial_vol, dtype=float))
    rate = np.broadcast_to(_as_array(rate), spot.shape)
    price = np.broadcast_to(price, spot.shape)
    option_type = _broadcast_option_type(option_type, spot.shape)
    initial_sigma = np.broadcast_to(np.maximum(_as_array(initial_vol), 1e-6), spot.shape)

    intrinsic_call = np.maximum(spot - strike * np.exp(-rate * ttm), 0.0)
    intrinsic_put = np.maximum(strike * np.exp(-rate * ttm) - spot, 0.0)
    intrinsic = np.where(option_type == "CALL", intrinsic_call, intrinsic_put)
    clipped_price = np.maximum(price, intrinsic + 1e-10)

    sigma = initial_sigma.astype(float, copy=True)
    low = np.full(spot.shape, 1e-6, dtype=float)
    high = np.full(spot.shape, 5.0, dtype=float)

    for _ in range(max_iter):
        model_price = black_scholes_price(spot, strike, ttm, rate, sigma, option_type)
        greeks = black_scholes_greeks(spot, strike, ttm, rate, sigma, option_type)
        diff = model_price - clipped_price
        converged = np.abs(diff) < tol
        if np.all(converged):
            break

        low = np.where(diff > 0.0, low, sigma)
        high = np.where(diff > 0.0, sigma, high)
        newton_step = sigma - diff / np.maximum(greeks["vega"], 1e-8)
        out_of_bounds = (newton_step <= low) | (newton_step >= high) | ~np.isfinite(newton_step)
        bisection_step = 0.5 * (low + high)
        sigma = np.where(converged, sigma, np.where(out_of_bounds, bisection_step, newton_step))

    return sigma


def load_deribit_with_iv(root_path, underlying="BTC_USDC-PERPETUAL", side=None, day_limit=None, columns=None):
    root = Path(root_path) / f"underlying={underlying}"
    day_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if day_limit is not None:
        day_dirs = day_dirs[-int(day_limit):]

    frames = []
    sides = [side] if side is not None else ["CALL", "PUT"]
    for day_dir in day_dirs:
        for current_side in sides:
            parquet_path = day_dir / f"side={current_side}" / "data.parquet"
            if parquet_path.exists():
                frames.append(pd.read_parquet(parquet_path, columns=columns))

    if not frames:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pd.concat(frames, ignore_index=True)


def inspect_deribit_schema(root_path, underlying="BTC_USDC-PERPETUAL", dt_day=None, side="CALL"):
    available_days = available_deribit_days(root_path, underlying=underlying)
    if not available_days:
        raise FileNotFoundError(f"No day partitions found for {underlying}")

    if dt_day is None:
        dt_day = available_days[-1]

    parquet_path = (
        Path(root_path)
        / f"underlying={underlying}"
        / f"dt_day={int(dt_day)}"
        / f"side={side}"
        / "data.parquet"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"No parquet file found at {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    rows = parquet_file.metadata.num_rows
    columns = pd.DataFrame(
        {
            "column": schema.names,
            "dtype": [str(schema.field(name).type) for name in schema.names],
        }
    )
    return {
        "parquet_path": parquet_path,
        "dt_day": int(dt_day),
        "side": side,
        "rows": int(rows),
        "columns": columns,
    }


def top_symbols_for_day(
    root_path,
    underlying="BTC_USDC-PERPETUAL",
    dt_day=None,
    top_n=10,
    sides=("CALL", "PUT"),
):
    symbol_df = load_deribit_day(
        root_path=root_path,
        underlying=underlying,
        dt_day=dt_day,
        columns=["symbol", "side", "ts"],
        sides=sides,
    )
    counts = (
        symbol_df.groupby(["symbol", "side"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .rename(columns={"size": "quote_count"})
        .reset_index(drop=True)
    )
    top_counts = counts.head(int(top_n)).copy()
    return top_counts, top_counts["symbol"].tolist()


def available_deribit_days(root_path, underlying="BTC_USDC-PERPETUAL"):
    root = Path(root_path) / f"underlying={underlying}"
    return sorted(int(path.name.split("=")[-1]) for path in root.iterdir() if path.is_dir())


def load_deribit_day(
    root_path,
    underlying="BTC_USDC-PERPETUAL",
    dt_day=None,
    columns=None,
    sides=("CALL", "PUT"),
    symbols=None,
    ts=None,
):
    if dt_day is None:
        available_days = available_deribit_days(root_path, underlying=underlying)
        if not available_days:
            raise FileNotFoundError(f"No day partitions found for {underlying}")
        dt_day = available_days[-1]

    day_path = Path(root_path) / f"underlying={underlying}" / f"dt_day={int(dt_day)}"
    frames = []
    for side in sides:
        parquet_path = day_path / f"side={side}" / "data.parquet"
        if parquet_path.exists():
            filters = [("ts", "==", int(ts))] if ts is not None else None
            frame = pd.read_parquet(parquet_path, columns=columns, filters=filters)
            if symbols is not None:
                frame = frame[frame["symbol"].isin(symbols)]
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No parquet files found under {day_path}")
    return pd.concat(frames, ignore_index=True)


def extract_daily_snapshots(
    root_path,
    underlying="BTC_USDC-PERPETUAL",
    day_limit=None,
    columns=None,
    max_rel_spread=0.2,
    min_quotes=20,
    symbols=None,
):
    available_days = available_deribit_days(root_path, underlying=underlying)
    if day_limit is None:
        selected_days = available_days
    else:
        selected_days = available_days[-int(day_limit):]
    snapshots = []

    for dt_day in selected_days:
        day_counts = load_deribit_day(
            root_path=root_path,
            underlying=underlying,
            dt_day=dt_day,
            columns=["ts", "symbol"],
            symbols=symbols,
        )
        if day_counts.empty:
            continue

        target_ts, _ = pick_dense_snapshot(day_counts, min_quotes=min_quotes)
        snapshot = load_deribit_day(
            root_path=root_path,
            underlying=underlying,
            dt_day=dt_day,
            columns=columns,
            symbols=symbols,
            ts=target_ts,
        )
        snapshot = filter_liquid_quotes(snapshot, max_rel_spread=max_rel_spread)
        if snapshot.empty:
            continue

        snapshot["snapshot_day"] = dt_day
        snapshots.append(snapshot)

    if not snapshots:
        raise ValueError("No snapshots were extracted")
    return pd.concat(snapshots, ignore_index=True)


def summarize_daily_atm_term_structure(snapshot_panel, target_log_moneyness=0.0):
    panel = snapshot_panel.copy()
    panel["distance_to_atm"] = (panel["log_moneyness"] - target_log_moneyness).abs()
    idx = panel.groupby(["snapshot_day", "side", "expiry_datetime"])["distance_to_atm"].idxmin()
    return panel.loc[idx].sort_values(["snapshot_day", "expiry_datetime", "side"])


def pick_dense_snapshot(df, min_quotes=20):
    if df.empty:
        raise ValueError("Cannot pick a dense snapshot from an empty DataFrame")

    counts = df.groupby("ts")["symbol"].nunique().sort_values(ascending=False)
    eligible = counts[counts >= min_quotes]
    if eligible.empty:
        target_ts = counts.index[0]
    else:
        target_ts = eligible.index[0]
    snapshot = df.loc[df["ts"] == target_ts].copy()
    return target_ts, snapshot


def filter_liquid_quotes(df, max_rel_spread=0.25, min_ttm=1.0 / 365.0, min_price=5.0):
    filtered = df.copy()
    filtered = filtered[np.isfinite(filtered["iv_mid"])]
    filtered = filtered[np.isfinite(filtered["mid_price"])]
    filtered = filtered[np.isfinite(filtered["und_mid"])]
    filtered = filtered[np.isfinite(filtered["log_moneyness"])]
    filtered = filtered[np.isfinite(filtered["ttm"])]
    filtered = filtered[filtered["mid_price"] >= min_price]
    filtered = filtered[filtered["ttm"] >= min_ttm]
    filtered = filtered[filtered["rel_spread"].abs() <= max_rel_spread]
    return filtered


def aggregate_surface_quotes(df, ttm_decimals=5, moneyness_decimals=3):
    grouped = df.assign(
        ttm_bucket=df["ttm"].round(ttm_decimals),
        log_m_bucket=df["log_moneyness"].round(moneyness_decimals),
    )
    aggregated = (
        grouped.groupby(["ttm_bucket", "log_m_bucket"], as_index=False)
        .agg(
            iv_mid=("iv_mid", "median"),
            und_mid=("und_mid", "median"),
            strike_price=("strike_price", "median"),
            quote_count=("symbol", "count"),
        )
        .rename(columns={"ttm_bucket": "ttm", "log_m_bucket": "log_moneyness"})
    )
    return aggregated.sort_values(["ttm", "log_moneyness"])


def build_surface_grid(surface_df, ttm_points=25, moneyness_points=41):
    clean = surface_df[["ttm", "log_moneyness", "iv_mid"]].dropna().copy()
    clean = clean[np.isfinite(clean["iv_mid"])]
    clean = clean.sort_values(["ttm", "log_moneyness"])

    if clean.empty:
        raise ValueError("Surface DataFrame is empty after cleaning")

    ttm_grid = np.linspace(clean["ttm"].min(), clean["ttm"].max(), ttm_points)
    log_moneyness_grid = np.linspace(clean["log_moneyness"].min(), clean["log_moneyness"].max(), moneyness_points)
    mesh_t, mesh_k = np.meshgrid(ttm_grid, log_moneyness_grid, indexing="ij")
    iv_grid = griddata(
        points=clean[["ttm", "log_moneyness"]].to_numpy(),
        values=clean["iv_mid"].to_numpy(),
        xi=(mesh_t, mesh_k),
        method="linear",
    )

    if np.isnan(iv_grid).any():
        nearest = griddata(
            points=clean[["ttm", "log_moneyness"]].to_numpy(),
            values=clean["iv_mid"].to_numpy(),
            xi=(mesh_t, mesh_k),
            method="nearest",
        )
        iv_grid = np.where(np.isnan(iv_grid), nearest, iv_grid)

    iv_grid = np.clip(iv_grid, 1e-4, 5.0)
    total_variance_grid = iv_grid**2 * mesh_t
    return SurfaceGrid(
        ttm_grid=ttm_grid,
        log_moneyness_grid=log_moneyness_grid,
        iv_grid=iv_grid,
        total_variance_grid=total_variance_grid,
    )


def dupire_local_vol(surface_grid, vol_floor=0.05, vol_cap=3.0):
    t = surface_grid.ttm_grid
    k = surface_grid.log_moneyness_grid
    w = np.maximum(surface_grid.total_variance_grid, 1e-8)
    dwdt = np.gradient(w, t, axis=0, edge_order=2)
    dwdk = np.gradient(w, k, axis=1, edge_order=2)
    d2wdk2 = np.gradient(dwdk, k, axis=1, edge_order=2)
    mesh_t, mesh_k = np.meshgrid(t, k, indexing="ij")

    denominator = (
        1.0
        - (mesh_k / w) * dwdk
        + 0.25 * (-0.25 - 1.0 / w + (mesh_k**2) / (w**2)) * dwdk**2
        + 0.5 * d2wdk2
    )
    local_variance = np.divide(dwdt, denominator, out=np.full_like(dwdt, np.nan), where=np.abs(denominator) > 1e-8)
    local_variance = np.clip(local_variance, vol_floor**2, vol_cap**2)
    local_vol_grid = np.sqrt(local_variance)
    return SurfaceGrid(
        ttm_grid=surface_grid.ttm_grid,
        log_moneyness_grid=surface_grid.log_moneyness_grid,
        iv_grid=surface_grid.iv_grid,
        total_variance_grid=surface_grid.total_variance_grid,
        local_vol_grid=local_vol_grid,
    )


def make_surface_interpolator(ttm_grid, log_moneyness_grid, values):
    return RegularGridInterpolator(
        (ttm_grid, log_moneyness_grid),
        values,
        bounds_error=False,
        fill_value=None,
    )


def local_vol_paths(spot, ttm, surface_grid, n_paths=10000, n_steps=100, rate=0.0, seed=7):
    if surface_grid.local_vol_grid is None:
        raise ValueError("surface_grid.local_vol_grid is required")

    rng = np.random.default_rng(seed)
    dt = float(ttm) / n_steps
    sqrt_dt = np.sqrt(dt)
    spots = np.full(n_paths, float(spot), dtype=float)
    interpolator = make_surface_interpolator(
        surface_grid.ttm_grid,
        surface_grid.log_moneyness_grid,
        surface_grid.local_vol_grid,
    )

    for step in range(n_steps):
        current_t = min((step + 1) * dt, float(ttm))
        log_moneyness = np.log(np.maximum(spots, EPSILON) / float(spot))
        points = np.column_stack([
            np.full(n_paths, current_t, dtype=float),
            log_moneyness,
        ])
        sigma = interpolator(points)
        sigma = np.where(np.isfinite(sigma), sigma, np.nanmedian(surface_grid.local_vol_grid))
        z = rng.standard_normal(n_paths)
        spots = spots * np.exp((rate - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z)

    return spots


def price_local_vol_mc(spot, strike, ttm, rate, option_type, surface_grid, n_paths=25000, n_steps=120, seed=7):
    terminal_spots = local_vol_paths(
        spot=spot,
        ttm=ttm,
        surface_grid=surface_grid,
        n_paths=n_paths,
        n_steps=n_steps,
        rate=rate,
        seed=seed,
    )
    if str(option_type).upper() == "CALL":
        payoff = np.maximum(terminal_spots - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal_spots, 0.0)
    return float(np.exp(-rate * ttm) * payoff.mean())


def attach_black_scholes_metrics(df, rate=0.0, iv_column="iv_mid"):
    result = df.copy()
    greeks = black_scholes_greeks(
        spot=result["und_mid"].to_numpy(),
        strike=result["strike_price"].to_numpy(),
        ttm=result["ttm"].to_numpy(),
        rate=rate,
        vol=result[iv_column].to_numpy(),
        option_type=result["side"].to_numpy(),
    )
    result["bs_price"] = black_scholes_price(
        spot=result["und_mid"].to_numpy(),
        strike=result["strike_price"].to_numpy(),
        ttm=result["ttm"].to_numpy(),
        rate=rate,
        vol=result[iv_column].to_numpy(),
        option_type=result["side"].to_numpy(),
    )
    result["bs_error"] = result["bs_price"] - result["mid_price"]
    for greek_name, greek_value in greeks.items():
        result[f"bs_{greek_name}"] = greek_value
    return result


def _interp_1d_with_flat_extrapolation(x_train, y_train, x_eval):
    x_eval = np.asarray(x_eval, dtype=float)
    curve = (
        pd.DataFrame({"x": np.asarray(x_train, dtype=float), "y": np.asarray(y_train, dtype=float)})
        .groupby("x", as_index=False)["y"]
        .median()
        .sort_values("x")
    )
    if curve.empty:
        raise ValueError("Interpolation curve is empty")
    if len(curve) == 1:
        return np.full(x_eval.shape, float(curve["y"].iloc[0]), dtype=float)
    return np.interp(
        x_eval,
        curve["x"].to_numpy(),
        curve["y"].to_numpy(),
        left=float(curve["y"].iloc[0]),
        right=float(curve["y"].iloc[-1]),
    )


def fit_flat_vol_model(train_df, estimator="median"):
    if train_df.empty:
        raise ValueError("Training DataFrame is empty")
    estimator = str(estimator).lower()
    if estimator == "mean":
        vol = float(train_df["iv_mid"].mean())
    else:
        vol = float(train_df["iv_mid"].median())
    return {"family": "flat_bs", "estimator": estimator, "vol": float(np.clip(vol, 1e-4, 5.0))}


def predict_flat_vol_model(test_df, model, rate=0.0):
    return black_scholes_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        vol=float(model["vol"]),
        option_type=test_df["side"].to_numpy(),
    )


def fit_term_structure_model(train_df, atm_band=0.03, ttm_round_days=0.5):
    if train_df.empty:
        raise ValueError("Training DataFrame is empty")
    curve_source = train_df.loc[train_df["log_moneyness"].abs() <= float(atm_band)].copy()
    if curve_source.empty:
        curve_source = train_df.copy()
    curve_source["ttm_bucket_days"] = (
        np.round(curve_source["ttm_days"].to_numpy() / float(ttm_round_days)) * float(ttm_round_days)
    )
    curve = (
        curve_source.groupby("ttm_bucket_days", as_index=False)["iv_mid"]
        .median()
        .sort_values("ttm_bucket_days")
    )
    return {
        "family": "term_bs",
        "atm_band": float(atm_band),
        "ttm_round_days": float(ttm_round_days),
        "ttm_bucket_days": curve["ttm_bucket_days"].to_numpy(),
        "iv_curve": curve["iv_mid"].to_numpy(),
    }


def predict_term_structure_model(test_df, model, rate=0.0):
    predicted_vol = _interp_1d_with_flat_extrapolation(
        model["ttm_bucket_days"],
        model["iv_curve"],
        test_df["ttm_days"].to_numpy(),
    )
    return black_scholes_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        vol=np.clip(predicted_vol, 1e-4, 5.0),
        option_type=test_df["side"].to_numpy(),
    )


def fit_surface_bs_model(
    train_df,
    ttm_decimals=5,
    moneyness_decimals=3,
    ttm_points=18,
    moneyness_points=25,
):
    if train_df.empty:
        raise ValueError("Training DataFrame is empty")
    surface_quotes = aggregate_surface_quotes(
        train_df,
        ttm_decimals=ttm_decimals,
        moneyness_decimals=moneyness_decimals,
    )
    surface_grid = build_surface_grid(
        surface_quotes,
        ttm_points=ttm_points,
        moneyness_points=moneyness_points,
    )
    return {
        "family": "surface_bs",
        "ttm_decimals": int(ttm_decimals),
        "moneyness_decimals": int(moneyness_decimals),
        "ttm_points": int(ttm_points),
        "moneyness_points": int(moneyness_points),
        "surface_grid": surface_grid,
    }


def _predict_surface_vol(test_df, surface_grid):
    interpolator = make_surface_interpolator(
        surface_grid.ttm_grid,
        surface_grid.log_moneyness_grid,
        surface_grid.iv_grid,
    )
    points = np.column_stack(
        [
            np.clip(
                test_df["ttm"].to_numpy(),
                surface_grid.ttm_grid.min(),
                surface_grid.ttm_grid.max(),
            ),
            np.clip(
                test_df["log_moneyness"].to_numpy(),
                surface_grid.log_moneyness_grid.min(),
                surface_grid.log_moneyness_grid.max(),
            ),
        ]
    )
    predicted_vol = np.asarray(interpolator(points), dtype=float)
    fill_value = float(np.nanmedian(surface_grid.iv_grid))
    predicted_vol = np.where(np.isfinite(predicted_vol), predicted_vol, fill_value)
    return np.clip(predicted_vol, 1e-4, 5.0)


def predict_surface_bs_model(test_df, model, rate=0.0):
    predicted_vol = _predict_surface_vol(test_df, model["surface_grid"])
    return black_scholes_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        vol=predicted_vol,
        option_type=test_df["side"].to_numpy(),
    )


def fit_local_vol_model(
    train_df,
    ttm_decimals=5,
    moneyness_decimals=3,
    ttm_points=18,
    moneyness_points=25,
    vol_floor=0.08,
    vol_cap=1.25,
    n_paths=1200,
    n_steps=40,
):
    surface_model = fit_surface_bs_model(
        train_df,
        ttm_decimals=ttm_decimals,
        moneyness_decimals=moneyness_decimals,
        ttm_points=ttm_points,
        moneyness_points=moneyness_points,
    )
    local_surface = dupire_local_vol(
        surface_model["surface_grid"],
        vol_floor=vol_floor,
        vol_cap=vol_cap,
    )
    return {
        "family": "local_vol",
        "ttm_decimals": int(ttm_decimals),
        "moneyness_decimals": int(moneyness_decimals),
        "ttm_points": int(ttm_points),
        "moneyness_points": int(moneyness_points),
        "vol_floor": float(vol_floor),
        "vol_cap": float(vol_cap),
        "n_paths": int(n_paths),
        "n_steps": int(n_steps),
        "surface_grid": local_surface,
    }


def predict_local_vol_model(test_df, model, rate=0.0, seed=7):
    predictions = []
    for row_index, (_, row) in enumerate(test_df.iterrows()):
        predictions.append(
            price_local_vol_mc(
                spot=row["und_mid"],
                strike=row["strike_price"],
                ttm=row["ttm"],
                rate=rate,
                option_type=row["side"],
                surface_grid=model["surface_grid"],
                n_paths=model["n_paths"],
                n_steps=model["n_steps"],
                seed=int(seed + row_index),
            )
        )
    return np.asarray(predictions, dtype=float)


def _prepare_calibration_sample(train_df, max_quotes=120):
    calibration_df = train_df[
        [
            "snapshot_day",
            "side",
            "strike_price",
            "und_mid",
            "ttm",
            "ttm_days",
            "mid_price",
            "log_moneyness",
            "rel_spread",
        ]
    ].dropna().copy()
    if len(calibration_df) <= int(max_quotes):
        return calibration_df.reset_index(drop=True)

    ttm_bins = min(4, calibration_df["ttm_days"].nunique())
    moneyness_bins = min(5, calibration_df["log_moneyness"].nunique())
    calibration_df["ttm_bucket"] = pd.qcut(
        calibration_df["ttm_days"].rank(method="first"),
        q=max(ttm_bins, 1),
        labels=False,
        duplicates="drop",
    )
    calibration_df["moneyness_bucket"] = pd.qcut(
        calibration_df["log_moneyness"].rank(method="first"),
        q=max(moneyness_bins, 1),
        labels=False,
        duplicates="drop",
    )
    grouped = (
        calibration_df.groupby(
            ["snapshot_day", "side", "ttm_bucket", "moneyness_bucket"],
            as_index=False,
        )
        .agg(
            strike_price=("strike_price", "median"),
            und_mid=("und_mid", "median"),
            ttm=("ttm", "median"),
            ttm_days=("ttm_days", "median"),
            mid_price=("mid_price", "median"),
            log_moneyness=("log_moneyness", "median"),
            rel_spread=("rel_spread", "median"),
        )
        .sort_values(["rel_spread", "ttm_days", "log_moneyness"])
        .head(int(max_quotes))
    )
    return grouped.reset_index(drop=True)


def _heston_log_cf(phi, spot, ttm, rate, kappa, theta, sigma_v, rho, v0):
    phi = np.asarray(phi, dtype=np.complex128)
    sigma_v = max(float(sigma_v), EPSILON)
    kappa = max(float(kappa), EPSILON)
    theta = max(float(theta), EPSILON)
    rho = float(np.clip(rho, -0.999, 0.999))
    v0 = max(float(v0), EPSILON)
    spot = max(float(spot), EPSILON)
    ttm = max(float(ttm), EPSILON)
    rate = float(rate)

    x0 = np.log(spot)
    alpha = -0.5 * (phi * phi + 1j * phi)
    beta = kappa - rho * sigma_v * 1j * phi
    gamma = 0.5 * sigma_v * sigma_v
    d = np.sqrt(beta * beta - 4.0 * alpha * gamma)
    r_minus = (beta - d) / (sigma_v * sigma_v)
    r_plus = (beta + d) / (sigma_v * sigma_v)
    g = r_minus / np.where(np.abs(r_plus) < EPSILON, EPSILON + 0j, r_plus)
    exp_neg_dt = np.exp(-d * ttm)
    log_term = np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    C = kappa * theta * (r_minus * ttm - (2.0 / (sigma_v * sigma_v)) * log_term)
    D = r_minus * (1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt)
    return np.exp(C + D * v0 + 1j * phi * (x0 + rate * ttm))


def _bates_log_cf(phi, spot, ttm, rate, kappa, theta, sigma_v, rho, v0, jump_intensity, jump_mean, jump_vol):
    phi = np.asarray(phi, dtype=np.complex128)
    jump_intensity = max(float(jump_intensity), EPSILON)
    jump_mean = float(jump_mean)
    jump_vol = max(float(jump_vol), EPSILON)
    jump_kappa = np.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    base_cf = _heston_log_cf(
        phi,
        spot=spot,
        ttm=ttm,
        rate=rate,
        kappa=kappa,
        theta=theta,
        sigma_v=sigma_v,
        rho=rho,
        v0=v0,
    )
    jump_factor = np.exp(
        jump_intensity
        * ttm
        * (
            np.exp(1j * phi * jump_mean - 0.5 * jump_vol**2 * phi * phi)
            - 1.0
            - 1j * phi * jump_kappa
        )
    )
    return base_cf * jump_factor


def _call_price_from_log_cf(log_cf_func, spot, strike, ttm, rate, integration_limit=80.0, integration_points=160):
    spot = max(float(spot), EPSILON)
    strike = max(float(strike), EPSILON)
    ttm = max(float(ttm), EPSILON)
    rate = float(rate)
    integration_points = max(int(integration_points), 32)
    integration_limit = max(float(integration_limit), 10.0)

    u = np.linspace(1e-6, integration_limit, integration_points)
    log_strike = np.log(strike)
    cf_minus_i = log_cf_func(np.array([-1j], dtype=np.complex128))[0]
    cf_u = log_cf_func(u)
    cf_u_minus_i = log_cf_func(u - 1j)

    denominator_p1 = 1j * u * cf_minus_i
    denominator_p2 = 1j * u
    oscillation = np.exp(-1j * u * log_strike)

    integrand_p1 = np.real(oscillation * cf_u_minus_i / denominator_p1)
    integrand_p2 = np.real(oscillation * cf_u / denominator_p2)
    p1 = 0.5 + np.trapezoid(integrand_p1, u) / np.pi
    p2 = 0.5 + np.trapezoid(integrand_p2, u) / np.pi
    call_price = spot * p1 - strike * np.exp(-rate * ttm) * p2
    lower_bound = max(spot - strike * np.exp(-rate * ttm), 0.0)
    return float(max(call_price, lower_bound))


def _price_from_call(call_price, spot, strike, ttm, rate, option_type):
    if str(option_type).upper() == "CALL":
        return float(call_price)
    put_price = call_price - spot + strike * np.exp(-float(rate) * max(float(ttm), EPSILON))
    return float(max(put_price, 0.0))


def heston_price(
    spot,
    strike,
    ttm,
    rate,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    option_type,
    integration_limit=80.0,
    integration_points=160,
):
    spot_arr, strike_arr, ttm_arr, _ = _sanitize_inputs(spot, strike, ttm, np.ones_like(_as_array(ttm)))
    rate_arr = np.broadcast_to(_as_array(rate), spot_arr.shape)
    option_type_arr = _broadcast_option_type(option_type, spot_arr.shape)
    prices = []
    for current_spot, current_strike, current_ttm, current_rate, current_type in zip(
        spot_arr.ravel(),
        strike_arr.ravel(),
        ttm_arr.ravel(),
        rate_arr.ravel(),
        option_type_arr.ravel(),
    ):
        log_cf = lambda phi: _heston_log_cf(
            phi,
            spot=current_spot,
            ttm=current_ttm,
            rate=current_rate,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
        )
        call_price = _call_price_from_log_cf(
            log_cf,
            current_spot,
            current_strike,
            current_ttm,
            current_rate,
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        prices.append(_price_from_call(call_price, current_spot, current_strike, current_ttm, current_rate, current_type))
    return np.asarray(prices, dtype=float).reshape(spot_arr.shape)


def bates_price(
    spot,
    strike,
    ttm,
    rate,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    jump_intensity,
    jump_mean,
    jump_vol,
    option_type,
    integration_limit=80.0,
    integration_points=160,
):
    spot_arr, strike_arr, ttm_arr, _ = _sanitize_inputs(spot, strike, ttm, np.ones_like(_as_array(ttm)))
    rate_arr = np.broadcast_to(_as_array(rate), spot_arr.shape)
    option_type_arr = _broadcast_option_type(option_type, spot_arr.shape)
    prices = []
    for current_spot, current_strike, current_ttm, current_rate, current_type in zip(
        spot_arr.ravel(),
        strike_arr.ravel(),
        ttm_arr.ravel(),
        rate_arr.ravel(),
        option_type_arr.ravel(),
    ):
        log_cf = lambda phi: _bates_log_cf(
            phi,
            spot=current_spot,
            ttm=current_ttm,
            rate=current_rate,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_vol=jump_vol,
        )
        call_price = _call_price_from_log_cf(
            log_cf,
            current_spot,
            current_strike,
            current_ttm,
            current_rate,
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        prices.append(_price_from_call(call_price, current_spot, current_strike, current_ttm, current_rate, current_type))
    return np.asarray(prices, dtype=float).reshape(spot_arr.shape)


def _bounded_initial_guess(initial_guess, bounds):
    clipped = []
    for value, (lower, upper) in zip(initial_guess, bounds):
        clipped.append(float(np.clip(value, lower + 1e-8, upper - 1e-8)))
    return np.asarray(clipped, dtype=float)


def _calibration_loss(market_price, model_price, scale):
    error = (np.asarray(model_price, dtype=float) - np.asarray(market_price, dtype=float)) / scale
    return float(np.mean(error**2))


def fit_heston_model(
    train_df,
    initial_guess=(1.5, 0.04, 0.35, -0.6, 0.04),
    bounds=((0.2, 8.0), (0.01, 0.80), (0.05, 2.00), (-0.95, -0.05), (0.01, 0.80)),
    integration_limit=80.0,
    integration_points=160,
    maxiter=25,
    max_quotes=120,
):
    calibration_df = _prepare_calibration_sample(train_df, max_quotes=max_quotes)
    scale = np.maximum(calibration_df["mid_price"].to_numpy(), 25.0)
    initial_guess = _bounded_initial_guess(initial_guess, bounds)

    def objective(params):
        kappa, theta, sigma_v, rho, v0 = params
        penalty = 0.0
        if 2.0 * kappa * theta <= sigma_v * sigma_v:
            penalty += 0.1 * (sigma_v * sigma_v - 2.0 * kappa * theta) ** 2
        model_price = heston_price(
            spot=calibration_df["und_mid"].to_numpy(),
            strike=calibration_df["strike_price"].to_numpy(),
            ttm=calibration_df["ttm"].to_numpy(),
            rate=0.0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            option_type=calibration_df["side"].to_numpy(),
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        return _calibration_loss(calibration_df["mid_price"].to_numpy(), model_price, scale) + penalty

    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    fitted_params = result.x if result.success else initial_guess
    return {
        "family": "heston",
        "kappa": float(fitted_params[0]),
        "theta": float(fitted_params[1]),
        "sigma_v": float(fitted_params[2]),
        "rho": float(fitted_params[3]),
        "v0": float(fitted_params[4]),
        "integration_limit": float(integration_limit),
        "integration_points": int(integration_points),
        "maxiter": int(maxiter),
        "max_quotes": int(max_quotes),
        "objective": float(result.fun) if hasattr(result, "fun") else np.nan,
        "success": bool(getattr(result, "success", False)),
    }


def predict_heston_model(test_df, model, rate=0.0):
    return heston_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        kappa=model["kappa"],
        theta=model["theta"],
        sigma_v=model["sigma_v"],
        rho=model["rho"],
        v0=model["v0"],
        option_type=test_df["side"].to_numpy(),
        integration_limit=model["integration_limit"],
        integration_points=model["integration_points"],
    )


def fit_bates_model(
    train_df,
    initial_guess=(1.5, 0.04, 0.35, -0.6, 0.04, 0.40, -0.05, 0.12),
    bounds=((0.2, 8.0), (0.01, 0.80), (0.05, 2.00), (-0.95, -0.05), (0.01, 0.80), (0.05, 2.00), (-0.30, 0.05), (0.03, 0.40)),
    integration_limit=80.0,
    integration_points=160,
    maxiter=30,
    max_quotes=120,
):
    calibration_df = _prepare_calibration_sample(train_df, max_quotes=max_quotes)
    scale = np.maximum(calibration_df["mid_price"].to_numpy(), 25.0)
    initial_guess = _bounded_initial_guess(initial_guess, bounds)

    def objective(params):
        kappa, theta, sigma_v, rho, v0, jump_intensity, jump_mean, jump_vol = params
        penalty = 0.0
        if 2.0 * kappa * theta <= sigma_v * sigma_v:
            penalty += 0.1 * (sigma_v * sigma_v - 2.0 * kappa * theta) ** 2
        model_price = bates_price(
            spot=calibration_df["und_mid"].to_numpy(),
            strike=calibration_df["strike_price"].to_numpy(),
            ttm=calibration_df["ttm"].to_numpy(),
            rate=0.0,
            kappa=kappa,
            theta=theta,
            sigma_v=sigma_v,
            rho=rho,
            v0=v0,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_vol=jump_vol,
            option_type=calibration_df["side"].to_numpy(),
            integration_limit=integration_limit,
            integration_points=integration_points,
        )
        return _calibration_loss(calibration_df["mid_price"].to_numpy(), model_price, scale) + penalty

    result = minimize(objective, initial_guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
    fitted_params = result.x if result.success else initial_guess
    return {
        "family": "bates",
        "kappa": float(fitted_params[0]),
        "theta": float(fitted_params[1]),
        "sigma_v": float(fitted_params[2]),
        "rho": float(fitted_params[3]),
        "v0": float(fitted_params[4]),
        "jump_intensity": float(fitted_params[5]),
        "jump_mean": float(fitted_params[6]),
        "jump_vol": float(fitted_params[7]),
        "integration_limit": float(integration_limit),
        "integration_points": int(integration_points),
        "maxiter": int(maxiter),
        "max_quotes": int(max_quotes),
        "objective": float(result.fun) if hasattr(result, "fun") else np.nan,
        "success": bool(getattr(result, "success", False)),
    }


def predict_bates_model(test_df, model, rate=0.0):
    return bates_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        kappa=model["kappa"],
        theta=model["theta"],
        sigma_v=model["sigma_v"],
        rho=model["rho"],
        v0=model["v0"],
        jump_intensity=model["jump_intensity"],
        jump_mean=model["jump_mean"],
        jump_vol=model["jump_vol"],
        option_type=test_df["side"].to_numpy(),
        integration_limit=model["integration_limit"],
        integration_points=model["integration_points"],
    )


def _discounted_lognormal_option_price(log_mean, log_var, strike, rate, ttm, option_type):
    strike = np.maximum(_as_array(strike), EPSILON)
    log_var = np.maximum(_as_array(log_var), EPSILON)
    log_std = np.sqrt(log_var)
    d2 = (log_mean - np.log(strike)) / log_std
    d1 = d2 + log_std
    expected_spot = np.exp(log_mean + 0.5 * log_var)
    discount = np.exp(-_as_array(rate) * _as_array(ttm))
    call = discount * (expected_spot * norm.cdf(d1) - strike * norm.cdf(d2))
    put = discount * (strike * norm.cdf(-d2) - expected_spot * norm.cdf(-d1))
    option_type = _broadcast_option_type(option_type, call.shape)
    return np.where(option_type == "CALL", call, put)


def merton_jump_price(
    spot,
    strike,
    ttm,
    rate,
    diffusion_vol,
    jump_intensity,
    jump_mean,
    jump_vol,
    option_type,
    max_terms=40,
):
    spot, strike, ttm, diffusion_vol = _sanitize_inputs(spot, strike, ttm, diffusion_vol)
    rate = np.broadcast_to(_as_array(rate), spot.shape)
    jump_intensity = np.broadcast_to(np.maximum(_as_array(jump_intensity), EPSILON), spot.shape)
    jump_mean = np.broadcast_to(_as_array(jump_mean), spot.shape)
    jump_vol = np.broadcast_to(np.maximum(_as_array(jump_vol), EPSILON), spot.shape)
    option_type = _broadcast_option_type(option_type, spot.shape)

    jump_kappa = np.exp(jump_mean + 0.5 * jump_vol**2) - 1.0
    lambda_t = jump_intensity * ttm
    total_price = np.zeros(spot.shape, dtype=float)

    for n in range(int(max_terms)):
        if n == 0:
            weight = np.exp(-lambda_t)
        else:
            safe_lambda_t = np.maximum(lambda_t, EPSILON)
            log_weight = -lambda_t + n * np.log(safe_lambda_t) - gammaln(n + 1)
            weight = np.where(lambda_t > 0.0, np.exp(log_weight), 0.0)

        conditional_log_mean = (
            np.log(spot)
            + (rate - jump_intensity * jump_kappa - 0.5 * diffusion_vol**2) * ttm
            + n * jump_mean
        )
        conditional_log_var = diffusion_vol**2 * ttm + n * jump_vol**2
        conditional_price = _discounted_lognormal_option_price(
            conditional_log_mean,
            conditional_log_var,
            strike,
            rate,
            ttm,
            option_type,
        )
        total_price += weight * conditional_price

    return total_price


def fit_merton_jump_model(
    train_df,
    jump_intensity,
    jump_mean,
    jump_vol,
    atm_band=0.03,
    estimator="median",
    max_terms=40,
):
    if train_df.empty:
        raise ValueError("Training DataFrame is empty")
    estimator = str(estimator).lower()
    curve_source = train_df.loc[train_df["log_moneyness"].abs() <= float(atm_band)].copy()
    if curve_source.empty:
        curve_source = train_df.copy()
    if estimator == "mean":
        diffusion_vol = float(curve_source["iv_mid"].mean())
    else:
        diffusion_vol = float(curve_source["iv_mid"].median())
    return {
        "family": "merton_jump",
        "jump_intensity": float(max(jump_intensity, EPSILON)),
        "jump_mean": float(jump_mean),
        "jump_vol": float(max(jump_vol, EPSILON)),
        "atm_band": float(atm_band),
        "estimator": estimator,
        "diffusion_vol": float(np.clip(diffusion_vol, 1e-4, 5.0)),
        "max_terms": int(max_terms),
    }


def predict_merton_jump_model(test_df, model, rate=0.0):
    return merton_jump_price(
        spot=test_df["und_mid"].to_numpy(),
        strike=test_df["strike_price"].to_numpy(),
        ttm=test_df["ttm"].to_numpy(),
        rate=rate,
        diffusion_vol=float(model["diffusion_vol"]),
        jump_intensity=float(model["jump_intensity"]),
        jump_mean=float(model["jump_mean"]),
        jump_vol=float(model["jump_vol"]),
        option_type=test_df["side"].to_numpy(),
        max_terms=int(model["max_terms"]),
    )


def summarize_pricing_errors(actual_price, predicted_price):
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    error = predicted_price - actual_price
    abs_error = np.abs(error)
    return {
        "quote_count": int(len(error)),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(error.mean()),
        "median_abs_error": float(np.median(abs_error)),
        "sum_abs_error": float(abs_error.sum()),
        "sum_sq_error": float(np.square(error).sum()),
        "sum_error": float(error.sum()),
    }


def fit_pricing_model(train_df, family, config):
    config = dict(config)
    if family == "flat_bs":
        return fit_flat_vol_model(train_df, estimator=config.get("estimator", "median"))
    if family == "term_bs":
        return fit_term_structure_model(
            train_df,
            atm_band=config.get("atm_band", 0.03),
            ttm_round_days=config.get("ttm_round_days", 0.5),
        )
    if family == "surface_bs":
        return fit_surface_bs_model(
            train_df,
            ttm_decimals=config.get("ttm_decimals", 5),
            moneyness_decimals=config.get("moneyness_decimals", 3),
            ttm_points=config.get("ttm_points", 18),
            moneyness_points=config.get("moneyness_points", 25),
        )
    if family == "local_vol":
        return fit_local_vol_model(
            train_df,
            ttm_decimals=config.get("ttm_decimals", 5),
            moneyness_decimals=config.get("moneyness_decimals", 3),
            ttm_points=config.get("ttm_points", 18),
            moneyness_points=config.get("moneyness_points", 25),
            vol_floor=config.get("vol_floor", 0.08),
            vol_cap=config.get("vol_cap", 1.25),
            n_paths=config.get("n_paths", 1200),
            n_steps=config.get("n_steps", 40),
        )
    if family == "merton_jump":
        return fit_merton_jump_model(
            train_df,
            jump_intensity=config.get("jump_intensity", 0.5),
            jump_mean=config.get("jump_mean", -0.05),
            jump_vol=config.get("jump_vol", 0.15),
            atm_band=config.get("atm_band", 0.03),
            estimator=config.get("estimator", "median"),
            max_terms=config.get("max_terms", 40),
        )
    if family == "heston":
        return fit_heston_model(
            train_df,
            initial_guess=config.get("initial_guess", (1.5, 0.04, 0.35, -0.6, 0.04)),
            bounds=config.get("bounds", ((0.2, 8.0), (0.01, 0.80), (0.05, 2.00), (-0.95, -0.05), (0.01, 0.80))),
            integration_limit=config.get("integration_limit", 80.0),
            integration_points=config.get("integration_points", 160),
            maxiter=config.get("maxiter", 25),
            max_quotes=config.get("max_quotes", 120),
        )
    if family == "bates":
        return fit_bates_model(
            train_df,
            initial_guess=config.get("initial_guess", (1.5, 0.04, 0.35, -0.6, 0.04, 0.40, -0.05, 0.12)),
            bounds=config.get("bounds", ((0.2, 8.0), (0.01, 0.80), (0.05, 2.00), (-0.95, -0.05), (0.01, 0.80), (0.05, 2.00), (-0.30, 0.05), (0.03, 0.40))),
            integration_limit=config.get("integration_limit", 80.0),
            integration_points=config.get("integration_points", 160),
            maxiter=config.get("maxiter", 30),
            max_quotes=config.get("max_quotes", 120),
        )
    raise ValueError(f"Unknown pricing model family: {family}")


def predict_pricing_model(test_df, model, rate=0.0, seed=7):
    family = model["family"]
    if family == "flat_bs":
        return predict_flat_vol_model(test_df, model, rate=rate)
    if family == "term_bs":
        return predict_term_structure_model(test_df, model, rate=rate)
    if family == "surface_bs":
        return predict_surface_bs_model(test_df, model, rate=rate)
    if family == "local_vol":
        return predict_local_vol_model(test_df, model, rate=rate, seed=seed)
    if family == "merton_jump":
        return predict_merton_jump_model(test_df, model, rate=rate)
    if family == "heston":
        return predict_heston_model(test_df, model, rate=rate)
    if family == "bates":
        return predict_bates_model(test_df, model, rate=rate)
    raise ValueError(f"Unknown fitted pricing model family: {family}")


def _aggregate_error_rows(error_rows):
    error_frame = pd.DataFrame(error_rows)
    total_quotes = int(error_frame["quote_count"].sum())
    return {
        "days": int(error_frame["day"].nunique()),
        "quote_count": total_quotes,
        "mae": float(error_frame["sum_abs_error"].sum() / total_quotes),
        "rmse": float(np.sqrt(error_frame["sum_sq_error"].sum() / total_quotes)),
        "bias": float(error_frame["sum_error"].sum() / total_quotes),
        "median_daily_abs_error": float(error_frame["median_abs_error"].median()),
    }


def walk_forward_model_grid_search(
    snapshot_panel,
    model_grids,
    rate=0.0,
    train_window_days=5,
    train_fraction=0.7,
    seed=7,
):
    if snapshot_panel.empty:
        raise ValueError("Snapshot panel is empty")

    unique_days = np.asarray(sorted(snapshot_panel["snapshot_day"].unique()), dtype=int)
    if len(unique_days) <= train_window_days + 1:
        raise ValueError("Not enough daily snapshots for walk-forward evaluation")

    split_idx = max(train_window_days + 1, int(np.ceil(len(unique_days) * float(train_fraction))))
    split_idx = min(split_idx, len(unique_days) - 1)
    if split_idx <= train_window_days:
        raise ValueError("Training split leaves no validation days for the grid search")

    validation_rows = []
    best_config_rows = []
    best_configs = {}

    for family, configs in model_grids.items():
        family_best_row = None
        for config in configs:
            error_rows = []
            for day_position in range(train_window_days, split_idx):
                fit_days = unique_days[day_position - train_window_days:day_position]
                validation_day = int(unique_days[day_position])
                train_df = snapshot_panel.loc[snapshot_panel["snapshot_day"].isin(fit_days)].copy()
                validation_df = snapshot_panel.loc[snapshot_panel["snapshot_day"] == validation_day].copy()
                if train_df.empty or validation_df.empty:
                    continue
                try:
                    model = fit_pricing_model(train_df, family, config)
                    predicted_price = predict_pricing_model(
                        validation_df,
                        model,
                        rate=rate,
                        seed=int(seed + validation_day),
                    )
                    metrics = summarize_pricing_errors(validation_df["mid_price"].to_numpy(), predicted_price)
                except Exception:
                    continue
                metrics["day"] = validation_day
                error_rows.append(metrics)

            if not error_rows:
                continue

            aggregate = _aggregate_error_rows(error_rows)
            result_row = {
                "family": family,
                "config": repr(config),
                "config_dict": dict(config),
                "validation_days": aggregate["days"],
                "validation_quotes": aggregate["quote_count"],
                "validation_mae": aggregate["mae"],
                "validation_rmse": aggregate["rmse"],
                "validation_bias": aggregate["bias"],
                "validation_median_daily_abs_error": aggregate["median_daily_abs_error"],
            }
            validation_rows.append(result_row)

            if family_best_row is None or result_row["validation_mae"] < family_best_row["validation_mae"]:
                family_best_row = result_row

        if family_best_row is not None:
            best_configs[family] = family_best_row["config_dict"]
            best_config_rows.append(family_best_row)

    if not best_configs:
        raise ValueError("Grid search did not produce a valid pricing model")

    test_daily_rows = []
    for family, best_config in best_configs.items():
        for day_position in range(split_idx, len(unique_days)):
            fit_days = unique_days[max(0, day_position - train_window_days):day_position]
            test_day = int(unique_days[day_position])
            train_df = snapshot_panel.loc[snapshot_panel["snapshot_day"].isin(fit_days)].copy()
            test_df = snapshot_panel.loc[snapshot_panel["snapshot_day"] == test_day].copy()
            if train_df.empty or test_df.empty:
                continue
            try:
                model = fit_pricing_model(train_df, family, best_config)
                predicted_price = predict_pricing_model(
                    test_df,
                    model,
                    rate=rate,
                    seed=int(seed + test_day),
                )
                metrics = summarize_pricing_errors(test_df["mid_price"].to_numpy(), predicted_price)
            except Exception:
                continue
            metrics.update(
                {
                    "family": family,
                    "config": repr(best_config),
                    "day": test_day,
                }
            )
            test_daily_rows.append(metrics)

    test_daily = pd.DataFrame(test_daily_rows)
    if test_daily.empty:
        raise ValueError("Walk-forward test evaluation produced no rows")

    test_summary_rows = []
    for family, family_frame in test_daily.groupby("family"):
        aggregate = _aggregate_error_rows(family_frame.to_dict("records"))
        test_summary_rows.append(
            {
                "family": family,
                "test_days": aggregate["days"],
                "test_quotes": aggregate["quote_count"],
                "test_mae": aggregate["mae"],
                "test_rmse": aggregate["rmse"],
                "test_bias": aggregate["bias"],
                "test_median_daily_abs_error": aggregate["median_daily_abs_error"],
            }
        )

    validation_grid = pd.DataFrame(validation_rows).sort_values(["validation_mae", "family"]).reset_index(drop=True)
    best_configs_df = pd.DataFrame(best_config_rows).sort_values("validation_mae").reset_index(drop=True)
    test_summary = pd.DataFrame(test_summary_rows).sort_values("test_mae").reset_index(drop=True)

    return {
        "train_days": unique_days[:split_idx].tolist(),
        "test_days": unique_days[split_idx:].tolist(),
        "validation_grid": validation_grid,
        "best_configs": best_configs_df,
        "test_daily": test_daily.sort_values(["family", "day"]).reset_index(drop=True),
        "test_summary": test_summary,
    }