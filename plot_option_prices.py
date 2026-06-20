"""Plot put/call option prices over time from a downloaded CSV.

Example:
    python plot_option_prices.py data/AAPL/put/all_puts.csv \
        --strike 180 --maturity 2024-03-15

Reads the CSV produced by download_option_data.py and plots the chosen
price field (default: mid) for a strike across maturities or, without
--strike, for all strikes across maturities.

If the CSV includes underlyingPrice, the plot overlays the underlying
security price on a secondary y-axis. When visible rows disagree for the
same timestamp, the line shows their mean and a shaded band shows the
visible min/max range.

If --strike is omitted, all strikes across the available maturities are
loaded. The selected maturity is visible by default, each maturity uses a
different color, checkboxes toggle maturity visibility, and a slider filters
the visible minimum and maximum strikes. When underlyingPrice is available,
that slider starts around 20% to 30% below the latest underlying price.

If --strike is provided, all maturities available for that strike are loaded
the same way, except there is no strike slider because only one strike is
shown.

The plot shows radio-button toggles for each available plot field so the
visible series can be switched without reloading the CSV.

The plot starts in Latest mode, which shows the values for the single global
latest quote timestamp against strike price on the x-axis. Historical mode
shows the existing time-series view. A title-row toggle switches between
those modes without reloading the CSV.

In Latest mode, a dashed secondary right-axis line is added for each visible
maturity in the same color as the primary line. That secondary field is loss
when the main field is not loss, or ask when the main field is loss. The
time-based underlying-price overlay remains available in Historical mode.

If the CSV path is omitted, it is resolved from environment variables
(loaded from .env if present):
    PLOT_CSV          explicit path to a CSV file (wins if set)
    OUT_DIR           data root, default ./data
    SYMBOL, SIDE      same vars used by download_option_data.py;
                      resolves to OUT_DIR/SYMBOL/SIDE/all_{SIDE}s.csv
    PLOT_FIELD        plot field: bid|mid|ask|last|loss (default: mid);
                      loss is put-only and shows protective-put loss %
                      under an 80% market drop using the row's ask price;
                      loss@n% uses n/100 as the post-crash underlying price,
                      so loss@30% means a 70% market drop
    INITIAL_STRIKE_MIN_PRICE_FRACTION
                      lower bound multiplier for the default strike slider
                      range without --strike (default: 0.7)
    INITIAL_STRIKE_MAX_PRICE_FRACTION
                      upper bound multiplier for the default strike slider
                      range without --strike (default: 0.8)

If --maturity is omitted, the latest maturity available for the selected
strike is used, or the latest maturity overall starts visible when
--strike is omitted.
"""

from __future__ import annotations

import argparse
from numbers import Real
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, RadioButtons, RangeSlider

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PLOT_FIELDS = ("bid", "mid", "ask", "last", "loss")
LOSS_FIELD = "loss"
UNDERLYING_PRICE_FIELD = "underlyingPrice"
CRASH_PRICE_FRACTION = 0.2
LOSS_FIELD_PATTERN = re.compile(r"^loss@(?P<percent>\d+(?:\.\d+)?)%$")
INITIAL_STRIKE_MIN_PRICE_FRACTION_ENV = "INITIAL_STRIKE_MIN_PRICE_FRACTION"
INITIAL_STRIKE_MAX_PRICE_FRACTION_ENV = "INITIAL_STRIKE_MAX_PRICE_FRACTION"
DEFAULT_INITIAL_STRIKE_MIN_PRICE_FRACTION = 0.7
DEFAULT_INITIAL_STRIKE_MAX_PRICE_FRACTION = 0.8
LOSS_FIELD_REQUIREMENTS = ("optionSymbol", "underlyingPrice", "strikePrice", "ask")
END_LABEL_X_MARGIN = 0.1
LATEST_MODE_X_MARGIN = 0.02
CONTROL_PANEL_WIDTH = 0.12
CONTROL_PANEL_LEFT = 0.86
CONTROL_PANEL_PLOT_RIGHT = 0.81
CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY = 0.76
CONTROL_PANEL_MATURITY_BOTTOM = 0.34
CONTROL_PANEL_DEFAULT_TOP = 0.9
STRIKE_SLIDER_PLOT_BOTTOM = 0.3
STRIKE_SLIDER_LEFT = 0.18
STRIKE_SLIDER_BOTTOM = 0.04
STRIKE_SLIDER_WIDTH = 0.64
STRIKE_SLIDER_HEIGHT = 0.04
STRIKE_SLIDER_LABEL = "Strike"
HISTORICAL_PLOT_MODE = "historical"
LATEST_PLOT_MODE = "latest"
MODE_TOGGLE_BUTTON_WIDTH = 0.07
MODE_TOGGLE_BUTTON_HEIGHT = 0.04
MODE_TOGGLE_BUTTON_GAP = 0.008
MODE_TOGGLE_VERTICAL_OFFSET = 0.01
MODE_TOGGLE_ACTIVE_COLOR = "0.85"
MODE_TOGGLE_INACTIVE_COLOR = "0.95"
MODE_TOGGLE_ACTIVE_TEXT_COLOR = "0.05"
MODE_TOGGLE_INACTIVE_TEXT_COLOR = "0.35"
XTICK_LABEL_FIGURE_LEFT_PADDING = 0.01
XTICK_LABEL_ROTATION = 30.0
MIN_PLOT_WIDTH = 0.2


@dataclass
class PlotSeries:
    line: Line2D
    quotes: pd.DataFrame
    maturity: str
    strike: float | None
    field: str = ""
    x_field: str = "updated"
    strike_values: tuple[float, ...] = ()


@dataclass
class PlotVisibility:
    visible_maturities: set[str]
    strike_bounds: tuple[float, float] | None = None
    mode: str = HISTORICAL_PLOT_MODE


@dataclass
class UnderlyingOverlay:
    ax: Axes
    line: Line2D
    band: PolyCollection | None = None


@dataclass
class SecondaryFieldOverlay:
    ax: Axes
    plotted_series: list[PlotSeries]
    field: str


@dataclass
class ModeToggle:
    latest_button: Button
    historical_button: Button


@dataclass(frozen=True)
class PlotFieldSpec:
    key: str
    label: str
    crash_price_fraction: float = CRASH_PRICE_FRACTION


@dataclass
class PlotContext:
    fig: Figure
    ax: Axes
    quotes: pd.DataFrame
    latest_quotes: pd.DataFrame | None
    title_prefix: str
    field: str
    visibility: PlotVisibility
    maturities: tuple[str, ...]
    maturity_colors: dict[str, str]
    plot_right: float
    plotted_series: list[PlotSeries]
    underlying_overlay: UnderlyingOverlay | None = None
    secondary_overlay: SecondaryFieldOverlay | None = None


def load_quotes(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["updated"])


def _missing_columns(quotes: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in columns if column not in quotes.columns)


def _format_loss_field_label(crash_price_fraction: float) -> str:
    percent = crash_price_fraction * 100.0
    return f"{LOSS_FIELD}@{percent:g}%"


def parse_plot_field(field: str) -> PlotFieldSpec:
    normalized = field.strip().lower()
    if normalized in PLOT_FIELDS:
        return PlotFieldSpec(key=normalized, label=normalized)

    match = LOSS_FIELD_PATTERN.fullmatch(normalized)
    if match is not None:
        percent = float(match.group("percent"))
        if not 0 < percent <= 100:
            raise SystemExit(
                "error: PLOT_FIELD loss percent must be greater than 0 and at most 100"
            )
        crash_price_fraction = percent / 100.0
        return PlotFieldSpec(
            key=LOSS_FIELD,
            label=_format_loss_field_label(crash_price_fraction),
            crash_price_fraction=crash_price_fraction,
        )

    raise SystemExit(
        f"error: PLOT_FIELD must be one of {PLOT_FIELDS} or {LOSS_FIELD}@n%, got {field!r}"
    )


def _compute_loss_field(
    quotes: pd.DataFrame,
    *,
    crash_price_fraction: float = CRASH_PRICE_FRACTION,
    field_label: str = LOSS_FIELD,
) -> pd.Series:
    underlying = pd.to_numeric(quotes["underlyingPrice"], errors="coerce")
    strike = pd.to_numeric(quotes["strikePrice"], errors="coerce")
    ask = pd.to_numeric(quotes["ask"], errors="coerce")

    if underlying.dropna().le(0).any():
        raise SystemExit(
            f"error: PLOT_FIELD={field_label!r} requires positive underlyingPrice values"
        )

    crash_price = underlying * crash_price_fraction
    protected_value = pd.concat([strike, crash_price], axis=1).max(axis=1)
    return ((underlying + ask - protected_value) / underlying) * 100.0


def _loss_field_error(quotes: pd.DataFrame, field_label: str = LOSS_FIELD) -> str | None:
    missing = _missing_columns(quotes, LOSS_FIELD_REQUIREMENTS)
    if missing:
        missing_list = ", ".join(missing)
        return f"error: PLOT_FIELD={field_label!r} requires CSV columns: {missing_list}"

    symbols = quotes["optionSymbol"].dropna().astype(str)
    if not symbols.empty and not symbols.str[-9].eq("P").all():
        return f"error: PLOT_FIELD={field_label!r} is only supported for put option data"

    underlying = pd.to_numeric(quotes["underlyingPrice"], errors="coerce")
    if underlying.dropna().le(0).any():
        return f"error: PLOT_FIELD={field_label!r} requires positive underlyingPrice values"

    return None


def enrich_quotes(
    quotes: pd.DataFrame,
    *,
    crash_price_fraction: float = CRASH_PRICE_FRACTION,
    overwrite_loss: bool = False,
    field_label: str = LOSS_FIELD,
) -> pd.DataFrame:
    if not overwrite_loss and LOSS_FIELD in quotes.columns:
        return quotes
    if _loss_field_error(quotes, field_label) is not None:
        return quotes

    prepared = quotes.copy()
    prepared[LOSS_FIELD] = _compute_loss_field(
        prepared,
        crash_price_fraction=crash_price_fraction,
        field_label=field_label,
    )
    return prepared


def load_prepared_quotes(
    csv_path: Path,
    *,
    crash_price_fraction: float = CRASH_PRICE_FRACTION,
    overwrite_loss: bool = False,
    field_label: str = LOSS_FIELD,
) -> pd.DataFrame:
    return enrich_quotes(
        load_quotes(csv_path),
        crash_price_fraction=crash_price_fraction,
        overwrite_loss=overwrite_loss,
        field_label=field_label,
    )


def _available_plot_field_keys(quotes: pd.DataFrame) -> tuple[str, ...]:
    return tuple(field for field in PLOT_FIELDS if field in quotes.columns)


def available_plot_fields(
    quotes: pd.DataFrame,
    *,
    loss_field_label: str = LOSS_FIELD,
) -> tuple[str, ...]:
    return tuple(
        loss_field_label if field == LOSS_FIELD else field
        for field in _available_plot_field_keys(quotes)
    )


def available_maturities(quotes: pd.DataFrame) -> tuple[str, ...]:
    maturities = quotes["maturityDate"].dropna().astype(str).unique().tolist()
    return tuple(sorted(maturities))


def _require_plot_field(quotes: pd.DataFrame, field: str) -> None:
    field_spec = parse_plot_field(field)
    if field_spec.key in _available_plot_field_keys(quotes):
        return
    if field_spec.key == LOSS_FIELD:
        error = _loss_field_error(quotes, field_spec.label)
        if error is not None:
            raise SystemExit(error)
    raise SystemExit(f"error: PLOT_FIELD={field_spec.label!r} is not available in CSV")


def prepare_quotes_for_field(quotes: pd.DataFrame, field: str) -> pd.DataFrame:
    field_spec = parse_plot_field(field)
    if field_spec.key == LOSS_FIELD:
        prepared = enrich_quotes(
            quotes,
            crash_price_fraction=field_spec.crash_price_fraction,
            overwrite_loss=True,
            field_label=field_spec.label,
        )
    else:
        prepared = enrich_quotes(quotes)
    _require_plot_field(prepared, field_spec.label)
    return prepared


def filter_to_latest_timestamp(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty or "updated" not in quotes.columns:
        return quotes

    latest_timestamp = quotes["updated"].max()
    if pd.isna(latest_timestamp):
        return quotes.iloc[0:0].copy()

    return quotes.loc[quotes["updated"] == latest_timestamp].sort_values(
        ["maturityDate", "strikePrice", "updated"]
    )


def _latest_mode_companion_field(field: str) -> str:
    return "ask" if parse_plot_field(field).key == LOSS_FIELD else LOSS_FIELD


def _plot_mode_xlabel(mode: str) -> str:
    if mode == LATEST_PLOT_MODE:
        return "Strike price"
    return "Quote date"


def _build_maturity_color_map(maturities: tuple[str, ...]) -> dict[str, str]:
    return {
        maturity: f"C{index % 10}"
        for index, maturity in enumerate(maturities)
    }


def _plot_right_for_quotes(quotes: pd.DataFrame, field: str) -> float:
    companion_field = _latest_mode_companion_field(field)
    companion_key = parse_plot_field(companion_field).key
    has_historical_overlay = UNDERLYING_PRICE_FIELD in quotes.columns
    has_latest_overlay = companion_key in _available_plot_field_keys(quotes)
    if has_historical_overlay or has_latest_overlay:
        return CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY
    return CONTROL_PANEL_PLOT_RIGHT


def _plot_quotes_for_mode(plot_context: PlotContext) -> pd.DataFrame:
    if plot_context.visibility.mode != LATEST_PLOT_MODE:
        return plot_context.quotes

    if plot_context.latest_quotes is None:
        plot_context.latest_quotes = filter_to_latest_timestamp(plot_context.quotes)
    return plot_context.latest_quotes


def _group_quotes_by_strike(quotes: pd.DataFrame) -> dict[float, pd.DataFrame]:
    grouped_quotes: dict[float, pd.DataFrame] = {}
    for strike, strike_quotes in quotes.groupby("strikePrice", sort=True):
        grouped_quotes[float(cast(Any, strike))] = strike_quotes
    return grouped_quotes


def filter_by_maturity(quotes: pd.DataFrame, maturity: str) -> pd.DataFrame:
    mask = quotes["maturityDate"] == maturity
    return quotes.loc[mask].sort_values(["strikePrice", "updated"])


def latest_maturity(quotes: pd.DataFrame, strike: float | None = None) -> str | None:
    df = quotes if strike is None else quotes.loc[quotes["strikePrice"] == strike]
    if df.empty:
        return None
    return df["maturityDate"].max()


def filter_by_strike(quotes: pd.DataFrame, strike: float | None) -> pd.DataFrame:
    if strike is None:
        return quotes
    return quotes.loc[quotes["strikePrice"] == strike].sort_values(["maturityDate", "updated"])


def _resolve_env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"error: {name} must be a float, got {raw_value!r}") from exc

    if value <= 0:
        raise SystemExit(f"error: {name} must be positive, got {raw_value!r}")

    return value


def _resolve_initial_strike_price_fractions() -> tuple[float, float]:
    lower = _resolve_env_float(
        INITIAL_STRIKE_MIN_PRICE_FRACTION_ENV,
        DEFAULT_INITIAL_STRIKE_MIN_PRICE_FRACTION,
    )
    upper = _resolve_env_float(
        INITIAL_STRIKE_MAX_PRICE_FRACTION_ENV,
        DEFAULT_INITIAL_STRIKE_MAX_PRICE_FRACTION,
    )
    if lower >= upper:
        raise SystemExit(
            "error: INITIAL_STRIKE_MIN_PRICE_FRACTION must be less than "
            "INITIAL_STRIKE_MAX_PRICE_FRACTION"
        )
    return lower, upper


def _snap_strike_bounds(
    strikes: list[float],
    target_bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    full_bounds = (strikes[0], strikes[-1])
    if target_bounds is None:
        return full_bounds

    lower_target, upper_target = sorted(target_bounds)
    lower_bound = next((strike for strike in strikes if strike >= lower_target), strikes[-1])
    upper_bound = next(
        (strike for strike in reversed(strikes) if strike <= upper_target),
        strikes[0],
    )
    lower_bound, upper_bound = sorted((lower_bound, upper_bound))
    return lower_bound, upper_bound


def _format_display_value(value: float) -> str:
    formatted = f"{value:.2f}".rstrip("0").rstrip(".")
    if formatted == "-0":
        return "0"
    return formatted


def _format_slider_percentage(value: float) -> str:
    return str(int(round(value)))


def _format_strike_range_text(
    bounds: tuple[float, float],
    reference_price: float | None = None,
) -> str:
    lower, upper = sorted(bounds)
    text = f"[{_format_display_value(lower)}, {_format_display_value(upper)}]"
    if reference_price is None or reference_price <= 0:
        return text

    lower_percentage = (lower / reference_price) * 100.0
    upper_percentage = (upper / reference_price) * 100.0
    return (
        f"{text} [{_format_slider_percentage(lower_percentage)}%-"
        f"{_format_slider_percentage(upper_percentage)}%]"
    )


def _set_strike_range_slider_text(
    slider: RangeSlider,
    bounds: tuple[float, float],
    reference_price: float | None = None,
) -> None:
    slider.valtext.set_text(_format_strike_range_text(bounds, reference_price))


def _last_underlying_price(quotes: pd.DataFrame) -> float | None:
    if "updated" not in quotes.columns or UNDERLYING_PRICE_FIELD not in quotes.columns:
        return None

    underlying_quotes = quotes.loc[:, ["updated", UNDERLYING_PRICE_FIELD]].copy()
    underlying_quotes[UNDERLYING_PRICE_FIELD] = pd.to_numeric(
        underlying_quotes[UNDERLYING_PRICE_FIELD],
        errors="coerce",
    )
    underlying_quotes = underlying_quotes.dropna(subset=["updated", UNDERLYING_PRICE_FIELD])
    if underlying_quotes.empty:
        return None

    latest_timestamp = underlying_quotes["updated"].max()
    latest_values = underlying_quotes.loc[
        underlying_quotes["updated"] == latest_timestamp,
        UNDERLYING_PRICE_FIELD,
    ]
    last_price = float(cast(Any, latest_values.mean()))
    if last_price <= 0:
        return None
    return last_price


def _default_strike_range_bounds(quotes: pd.DataFrame) -> tuple[float, float] | None:
    strikes = sorted(
        {float(cast(Any, value)) for value in quotes["strikePrice"].dropna().unique()}
    )
    if len(strikes) < 2:
        return None

    last_underlying_price = _last_underlying_price(quotes)
    if last_underlying_price is None:
        return None

    lower_fraction, upper_fraction = _resolve_initial_strike_price_fractions()

    return _snap_strike_bounds(
        strikes,
        (
            last_underlying_price * lower_fraction,
            last_underlying_price * upper_fraction,
        ),
    )


def plot_single_contract(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    label: str | None = None,
    *,
    x_field: str = "updated",
    **plot_kwargs: Any,
) -> Line2D:
    line_kwargs: dict[str, Any] = dict(plot_kwargs)
    line_kwargs.setdefault("marker", "o")
    line_kwargs.setdefault("markersize", 3)
    return ax.plot(quotes[x_field], quotes[field], label=label, **line_kwargs)[0]


def plot_all_strikes(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    *,
    color: str = "C0",
    label_prefix: str = "Strike",
    include_strike_in_label: bool = True,
    **plot_kwargs: Any,
) -> dict[float, Line2D]:
    lines_by_strike: dict[float, Line2D] = {}
    for strike, strike_quotes in _group_quotes_by_strike(quotes).items():
        label = label_prefix
        if include_strike_in_label:
            label = f"{label_prefix} {strike:g}"
        lines_by_strike[strike] = plot_single_contract(
            ax,
            strike_quotes,
            field,
            label=label,
            color=color,
            **plot_kwargs,
        )
    return lines_by_strike


def _build_historical_plot_series(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    *,
    maturity_colors: dict[str, str] | None = None,
    linestyle: str = "-",
    linewidth: float | None = None,
    alpha: float | None = None,
) -> list[PlotSeries]:
    plotted_series: list[PlotSeries] = []
    colors = maturity_colors or _build_maturity_color_map(available_maturities(quotes))
    overall_strike_count = len(_group_quotes_by_strike(quotes))
    for maturity in available_maturities(quotes):
        maturity_quotes = filter_by_maturity(quotes, maturity)
        grouped_quotes = _group_quotes_by_strike(maturity_quotes)
        line_kwargs: dict[str, Any] = {"linestyle": linestyle}
        if linewidth is not None:
            line_kwargs["linewidth"] = linewidth
        if alpha is not None:
            line_kwargs["alpha"] = alpha
        lines_by_strike = plot_all_strikes(
            ax,
            maturity_quotes,
            field,
            color=colors[maturity],
            label_prefix=(
                f"Expiry {maturity}"
                if overall_strike_count == 1
                else f"Expiry {maturity} · Strike"
            ),
            include_strike_in_label=overall_strike_count > 1,
            **line_kwargs,
        )
        for strike, line in lines_by_strike.items():
            plotted_series.append(
                PlotSeries(
                    line=line,
                    quotes=grouped_quotes[strike],
                    maturity=maturity,
                    strike=strike,
                    field=field,
                    x_field="updated",
                    strike_values=(strike,),
                )
            )
    return plotted_series


def _build_latest_plot_series(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    *,
    maturity_colors: dict[str, str] | None = None,
    linestyle: str = "-",
    linewidth: float | None = None,
    alpha: float | None = None,
) -> list[PlotSeries]:
    plotted_series: list[PlotSeries] = []
    colors = maturity_colors or _build_maturity_color_map(available_maturities(quotes))
    for maturity in available_maturities(quotes):
        maturity_quotes = filter_by_maturity(quotes, maturity).sort_values("strikePrice")
        line_kwargs: dict[str, Any] = {
            "color": colors[maturity],
            "linestyle": linestyle,
        }
        if linewidth is not None:
            line_kwargs["linewidth"] = linewidth
        if alpha is not None:
            line_kwargs["alpha"] = alpha
        line = plot_single_contract(
            ax,
            maturity_quotes,
            field,
            label=f"Expiry {maturity}",
            x_field="strikePrice",
            **line_kwargs,
        )
        strike_values = tuple(
            sorted(
                {
                    float(cast(Any, value))
                    for value in maturity_quotes["strikePrice"].dropna().unique()
                }
            )
        )
        plotted_series.append(
            PlotSeries(
                line=line,
                quotes=maturity_quotes,
                maturity=maturity,
                strike=None,
                field=field,
                x_field="strikePrice",
                strike_values=strike_values,
            )
        )
    return plotted_series


def build_plot_series(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    *,
    mode: str = HISTORICAL_PLOT_MODE,
    maturity_colors: dict[str, str] | None = None,
    linestyle: str = "-",
    linewidth: float | None = None,
    alpha: float | None = None,
) -> list[PlotSeries]:
    if mode == LATEST_PLOT_MODE:
        return _build_latest_plot_series(
            ax,
            quotes,
            field,
            maturity_colors=maturity_colors,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
    return _build_historical_plot_series(
        ax,
        quotes,
        field,
        maturity_colors=maturity_colors,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
    )


def _maturity_colors(plotted_series: list[PlotSeries]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for series in plotted_series:
        colors.setdefault(series.maturity, str(cast(Any, series.line.get_color())))
    return colors


def _visible_maturities(plotted_series: list[PlotSeries]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            series.maturity for series in plotted_series if series.line.get_visible()
        )
    )


def _series_last_y(series: PlotSeries) -> float:
    ydata = list(cast(Any, series.line.get_ydata()))
    if not ydata:
        return float("-inf")
    return float(cast(Any, ydata[-1]))


def _series_in_strike_bounds(
    series: PlotSeries,
    strike_bounds: tuple[float, float] | None,
) -> bool:
    if strike_bounds is None:
        return True
    series_strikes = _series_strikes(series)
    if not series_strikes:
        return True
    lower, upper = sorted(strike_bounds)
    return any(lower <= strike <= upper for strike in series_strikes)


def _series_strikes(series: PlotSeries) -> tuple[float, ...]:
    if series.strike_values:
        return series.strike_values
    if series.strike is not None:
        return (series.strike,)
    if "strikePrice" not in series.quotes.columns:
        return ()
    return tuple(
        sorted(
            {
                float(cast(Any, value))
                for value in series.quotes["strikePrice"].dropna().unique()
            }
        )
    )


def _can_refresh_series_from_quotes(series: PlotSeries) -> bool:
    return (
        bool(series.field)
        and series.x_field in series.quotes.columns
        and series.field in series.quotes.columns
    )


def _visible_quotes_for_series(
    series: PlotSeries,
    visibility: PlotVisibility,
) -> pd.DataFrame:
    if series.maturity not in visibility.visible_maturities:
        return series.quotes.iloc[0:0]

    visible_quotes = series.quotes
    if visibility.strike_bounds is None or "strikePrice" not in visible_quotes.columns:
        return visible_quotes

    lower, upper = sorted(visibility.strike_bounds)
    strike_values = pd.to_numeric(visible_quotes["strikePrice"], errors="coerce")
    return visible_quotes.loc[strike_values.between(lower, upper)]


def _visible_underlying_price_summary(
    plotted_series: list[PlotSeries],
    visibility: PlotVisibility | None = None,
) -> pd.DataFrame:
    visible_quotes = [
        series.quotes.loc[:, ["updated", UNDERLYING_PRICE_FIELD]]
        for series in plotted_series
        if series.line.get_visible() and UNDERLYING_PRICE_FIELD in series.quotes.columns
    ]
    if not visible_quotes and visibility is not None:
        visible_quotes = [
            series.quotes.loc[:, ["updated", UNDERLYING_PRICE_FIELD]]
            for series in plotted_series
            if UNDERLYING_PRICE_FIELD in series.quotes.columns
            and _series_in_strike_bounds(series, visibility.strike_bounds)
        ]

    if not visible_quotes:
        return pd.DataFrame(
            columns=[
                "updated",
                "underlyingPriceMean",
                "underlyingPriceMin",
                "underlyingPriceMax",
            ]
        )

    combined = pd.concat(visible_quotes, ignore_index=True)
    combined[UNDERLYING_PRICE_FIELD] = pd.to_numeric(
        combined[UNDERLYING_PRICE_FIELD],
        errors="coerce",
    )
    combined = combined.dropna(subset=["updated", UNDERLYING_PRICE_FIELD])
    if combined.empty:
        return pd.DataFrame(
            columns=[
                "updated",
                "underlyingPriceMean",
                "underlyingPriceMin",
                "underlyingPriceMax",
            ]
        )

    summary = (
        combined.groupby("updated", sort=True)[UNDERLYING_PRICE_FIELD]
        .agg(
            underlyingPriceMean="mean",
            underlyingPriceMin="min",
            underlyingPriceMax="max",
        )
        .reset_index()
        .sort_values("updated")
    )
    return summary


def _clear_underlying_overlay_band(overlay: UnderlyingOverlay) -> None:
    if overlay.band is None:
        return
    overlay.band.remove()
    overlay.band = None


def _update_underlying_overlay(
    overlay: UnderlyingOverlay,
    plotted_series: list[PlotSeries],
    visibility: PlotVisibility | None = None,
) -> None:
    summary = _visible_underlying_price_summary(plotted_series, visibility)
    _clear_underlying_overlay_band(overlay)

    if summary.empty:
        overlay.line.set_data([], [])
        overlay.ax.set_visible(False)
        return

    overlay.line.set_data(summary["updated"], summary["underlyingPriceMean"])
    overlay.ax.set_visible(True)

    if summary["underlyingPriceMax"].gt(summary["underlyingPriceMin"]).any():
        overlay.band = cast(
            PolyCollection,
            overlay.ax.fill_between(
                summary["updated"],
                summary["underlyingPriceMin"],
                summary["underlyingPriceMax"],
                color="0.2",
                alpha=0.15,
                linewidth=0,
            ),
        )

    lower = float(cast(Any, summary["underlyingPriceMin"].min()))
    upper = float(cast(Any, summary["underlyingPriceMax"].max()))
    padding = max((upper - lower) * 0.05, abs(upper) * 0.01, 0.01)
    overlay.ax.set_ylim(lower - padding, upper + padding)


def add_underlying_overlay(
    ax: Axes,
    plotted_series: list[PlotSeries],
) -> UnderlyingOverlay | None:
    if not any(UNDERLYING_PRICE_FIELD in series.quotes.columns for series in plotted_series):
        return None

    overlay_ax = ax.twinx()
    overlay_ax.set_ylabel("underlying price")
    overlay_ax.grid(False)
    overlay_ax.tick_params(axis="y", colors="0.25")
    line = overlay_ax.plot(
        [],
        [],
        color="0.25",
        linestyle="-",
        linewidth=2.5,
        alpha=0.85,
    )[0]
    overlay = UnderlyingOverlay(ax=overlay_ax, line=line)
    _update_underlying_overlay(overlay, plotted_series)
    return overlay


def add_latest_field_overlay(
    ax: Axes,
    quotes: pd.DataFrame,
    field: str,
    maturity_colors: dict[str, str],
) -> SecondaryFieldOverlay | None:
    field_spec = parse_plot_field(field)
    if field_spec.key not in _available_plot_field_keys(quotes):
        return None

    overlay_ax = ax.twinx()
    overlay_ax.set_ylabel(_field_ylabel(field_spec.label))
    overlay_ax.grid(False)
    overlay_ax.tick_params(axis="y", colors="0.25")
    _set_field_axis_direction(overlay_ax, field_spec.label)
    plotted_series = build_plot_series(
        overlay_ax,
        quotes,
        field_spec.key,
        mode=LATEST_PLOT_MODE,
        maturity_colors=maturity_colors,
        linestyle="--",
        linewidth=2,
        alpha=0.9,
    )
    if not plotted_series:
        overlay_ax.remove()
        return None
    return SecondaryFieldOverlay(
        ax=overlay_ax,
        plotted_series=plotted_series,
        field=field_spec.label,
    )


def _clear_strike_end_labels(fig: Figure) -> None:
    stored_labels = cast(Any, fig).__dict__.get("_strike_end_labels", [])
    for label in stored_labels:
        label.remove()
    _remember_control(fig, "_strike_end_labels", [])


def _sync_maturity_legend(ax: Axes, plotted_series: list[PlotSeries]) -> None:
    legend = cast(Any, ax).__dict__.get("legend_", None)
    if legend is not None:
        legend.remove()

    visible_maturities = _visible_maturities(plotted_series)
    if not visible_maturities:
        return

    colors = _maturity_colors(plotted_series)
    handles = [
        Line2D(
            [],
            [],
            color=colors[maturity],
            linestyle="-",
            marker="o",
            markersize=3,
            label=maturity,
        )
        for maturity in visible_maturities
    ]
    ax.legend(handles=handles, title="Maturity", loc="upper left")


def _sync_strike_end_labels(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
) -> None:
    _clear_strike_end_labels(fig)

    visible_series = [
        series for series in plotted_series if series.line.get_visible()
    ]
    if any(series.x_field == "strikePrice" for series in visible_series):
        return
    if len({series.strike for series in visible_series}) < 2:
        return

    labels = []
    for series in visible_series:
        xdata = list(cast(Any, series.line.get_xdata()))
        ydata = list(cast(Any, series.line.get_ydata()))
        if not xdata or not ydata:
            continue

        labels.append(
            ax.annotate(
                f"{series.strike:g}",
                xy=cast(Any, (xdata[-1], ydata[-1])),
                xytext=(6, 0),
                textcoords="offset points",
                color=str(cast(Any, series.line.get_color())),
                ha="left",
                va="center",
                annotation_clip=False,
                fontsize=8,
            )
        )

    _remember_control(fig, "_strike_end_labels", labels)


def _sync_maturity_guides(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
) -> None:
    if len({series.maturity for series in plotted_series}) < 2:
        _clear_strike_end_labels(fig)
        legend = cast(Any, ax).__dict__.get("legend_", None)
        if legend is not None:
            legend.remove()
        return

    _sync_maturity_legend(ax, plotted_series)
    _sync_strike_end_labels(fig, ax, plotted_series)


def _apply_visibility(
    ax: Axes,
    plotted_series: list[PlotSeries],
    visibility: PlotVisibility,
) -> None:
    ax.set_xmargin(
        LATEST_MODE_X_MARGIN
        if visibility.mode == LATEST_PLOT_MODE
        else END_LABEL_X_MARGIN
    )
    for series in plotted_series:
        if _can_refresh_series_from_quotes(series):
            visible_quotes = _visible_quotes_for_series(series, visibility)
            if visible_quotes.empty:
                series.line.set_data([], [])
                series.line.set_visible(False)
                continue
            series.line.set_data(
                visible_quotes[series.x_field],
                visible_quotes[series.field],
            )
            series.line.set_visible(True)
            continue

        is_visible = series.maturity in visibility.visible_maturities
        if visibility.strike_bounds is not None:
            is_visible = is_visible and _series_in_strike_bounds(
                series,
                visibility.strike_bounds,
            )
        series.line.set_visible(is_visible)
    ax.relim(visible_only=True)
    ax.autoscale_view()


def _visible_xtick_labels(ax: Axes) -> list[Any]:
    return [label for label in ax.get_xticklabels() if label.get_visible() and label.get_text()]


def _reset_xtick_label_alignment(ax: Axes) -> None:
    ax.tick_params(axis="x", labelrotation=XTICK_LABEL_ROTATION)
    for label in _visible_xtick_labels(ax):
        label.set_rotation(XTICK_LABEL_ROTATION)
        label.set_rotation_mode("anchor")
        label.set_ha("right")


def _reset_numeric_xtick_alignment(ax: Axes) -> None:
    ax.tick_params(axis="x", labelrotation=0)
    for label in _visible_xtick_labels(ax):
        label.set_rotation(0)
        label.set_rotation_mode("default")
        label.set_ha("center")


def _ensure_leftmost_xtick_visible(
    fig: Figure,
    ax: Axes,
    *,
    left_padding: float = XTICK_LABEL_FIGURE_LEFT_PADDING,
) -> None:
    _reset_xtick_label_alignment(ax)
    visible_labels = _visible_xtick_labels(ax)
    if not visible_labels:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leftmost_bbox = min(
        (
            label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            for label in visible_labels
        ),
        key=lambda bbox: bbox.x0,
    )
    overflow = left_padding - leftmost_bbox.x0
    if overflow <= 0:
        return

    plot_left, plot_right = _axis_box_x_bounds(ax)
    if plot_left is None or plot_right is None:
        return

    new_left = min(plot_left + overflow, plot_right - MIN_PLOT_WIDTH)
    if new_left <= plot_left:
        return

    fig.subplots_adjust(left=new_left)


def _realign_strike_slider(fig: Figure, ax: Axes) -> None:
    slider = cast(Any, fig).__dict__.get("_strike_range_slider")
    if slider is None:
        return

    slider_ax = getattr(slider, "ax", None)
    if slider_ax is None:
        return

    _align_slider_label_with_y_axis_labels(fig, ax, slider_ax, slider)


def _refresh_visibility(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
    visibility: PlotVisibility,
    underlying_overlay: UnderlyingOverlay | None = None,
    secondary_overlay: SecondaryFieldOverlay | None = None,
) -> None:
    _apply_visibility(ax, plotted_series, visibility)
    if secondary_overlay is not None:
        _apply_visibility(secondary_overlay.ax, secondary_overlay.plotted_series, visibility)
        secondary_overlay.ax.set_visible(
            any(series.line.get_visible() for series in secondary_overlay.plotted_series)
        )
    if underlying_overlay is not None:
        _update_underlying_overlay(underlying_overlay, plotted_series, visibility)
    if visibility.mode == LATEST_PLOT_MODE:
        _reset_numeric_xtick_alignment(ax)
    else:
        _ensure_leftmost_xtick_visible(fig, ax)
    _realign_strike_slider(fig, ax)
    _sync_maturity_guides(fig, ax, plotted_series)
    fig.canvas.draw_idle()


def _left_axis_tick_label_x0(fig: Figure, ax: Axes) -> float | None:
    renderer = fig.canvas.get_renderer()
    tick_bboxes = [
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        for label in ax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    if not tick_bboxes:
        return None
    return min(bbox.x0 for bbox in tick_bboxes)


def _axis_box_x_bounds(ax: Axes) -> tuple[float | None, float | None]:
    position = ax.get_position()
    left = getattr(position, "x0", None)
    right = getattr(position, "x1", None)
    if isinstance(left, Real) and isinstance(right, Real):
        return float(left), float(right)

    bounds = getattr(position, "bounds", None)
    if isinstance(bounds, tuple) and len(bounds) == 4:
        bound_left, _, bound_width, _ = bounds
        if isinstance(bound_left, Real) and isinstance(bound_width, Real):
            resolved_left = float(bound_left)
            return resolved_left, resolved_left + float(bound_width)

    return None, None


def _axis_box_y_bounds(ax: Axes) -> tuple[float | None, float | None]:
    position = ax.get_position()
    bottom = getattr(position, "y0", None)
    top = getattr(position, "y1", None)
    resolved_bottom = float(bottom) if isinstance(bottom, Real) else None
    resolved_top = float(top) if isinstance(top, Real) else None
    if resolved_bottom is not None or resolved_top is not None:
        return resolved_bottom, resolved_top

    bounds = getattr(position, "bounds", None)
    if isinstance(bounds, tuple) and len(bounds) == 4:
        _, bound_bottom, _, bound_height = bounds
        if isinstance(bound_bottom, Real) and isinstance(bound_height, Real):
            resolved_bottom = float(bound_bottom)
            return resolved_bottom, resolved_bottom + float(bound_height)

    return None, None


def _align_slider_label_with_y_axis_labels(
    fig: Figure,
    plot_ax: Axes,
    slider_ax: Axes,
    slider: RangeSlider,
    *,
    plot_right: float | None = None,
) -> None:
    label = slider.label
    if not hasattr(label, "get_window_extent"):
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bbox = label.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )
    label_left = getattr(label_bbox, "x0", None)
    if not isinstance(label_left, Real):
        return
    slider_bounds = slider_ax.get_position().bounds
    plot_left, current_plot_right = _axis_box_x_bounds(plot_ax)
    target_left = _left_axis_tick_label_x0(fig, plot_ax)
    if target_left is None:
        target_left = plot_left if plot_left is not None else slider_bounds[0]
    effective_plot_right = plot_right
    if effective_plot_right is None:
        effective_plot_right = current_plot_right
    if effective_plot_right is None:
        effective_plot_right = slider_bounds[0] + slider_bounds[2]
    shift = target_left - float(label_left)
    aligned_left = slider_bounds[0] + shift
    aligned_width = max(effective_plot_right - aligned_left, 0.01)
    slider_ax.set_position(
        (
            aligned_left,
            slider_bounds[1],
            aligned_width,
            slider_bounds[3],
        )
    )


def add_strike_range_slider(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
    visibility: PlotVisibility,
    underlying_overlay: UnderlyingOverlay | None = None,
    reference_price: float | None = None,
    *,
    plot_right: float | None = None,
    plot_context: PlotContext | None = None,
) -> RangeSlider | None:
    strikes = sorted(
        {
            strike
            for series in plotted_series
            for strike in _series_strikes(series)
        }
    )
    if len(strikes) < 2:
        return None

    visibility.strike_bounds = _snap_strike_bounds(strikes, visibility.strike_bounds)
    fig.subplots_adjust(bottom=STRIKE_SLIDER_PLOT_BOTTOM)
    slider_ax = fig.add_axes(
        (
            STRIKE_SLIDER_LEFT,
            STRIKE_SLIDER_BOTTOM,
            STRIKE_SLIDER_WIDTH,
            STRIKE_SLIDER_HEIGHT,
        )
    )
    slider = RangeSlider(
        ax=slider_ax,
        label=STRIKE_SLIDER_LABEL,
        valmin=strikes[0],
        valmax=strikes[-1],
        valinit=visibility.strike_bounds,
        valstep=strikes,
        valfmt="%0.0f",
    )
    _align_slider_label_with_y_axis_labels(
        fig,
        ax,
        slider_ax,
        slider,
        plot_right=plot_right,
    )
    _set_strike_range_slider_text(
        slider,
        visibility.strike_bounds,
        reference_price,
    )

    def update(bounds: tuple[float, float]) -> None:
        visibility.strike_bounds = (
            float(cast(Any, bounds[0])),
            float(cast(Any, bounds[1])),
        )
        _set_strike_range_slider_text(
            slider,
            visibility.strike_bounds,
            reference_price,
        )
        active_plotted_series = (
            plot_context.plotted_series if plot_context is not None else plotted_series
        )
        active_underlying_overlay = (
            plot_context.underlying_overlay if plot_context is not None else underlying_overlay
        )
        active_secondary_overlay = (
            plot_context.secondary_overlay if plot_context is not None else None
        )
        _refresh_visibility(
            fig,
            ax,
            active_plotted_series,
            visibility,
            active_underlying_overlay,
            active_secondary_overlay,
        )

    slider.on_changed(update)
    return slider


def add_maturity_toggle(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
    maturities: tuple[str, ...],
    visibility: PlotVisibility,
    underlying_overlay: UnderlyingOverlay | None = None,
    *,
    plot_right: float = CONTROL_PANEL_PLOT_RIGHT,
    maturity_colors: dict[str, str] | None = None,
    plot_context: PlotContext | None = None,
) -> CheckButtons | None:
    if len(maturities) < 2:
        return None

    fig.subplots_adjust(right=plot_right)
    _, plot_top = _axis_box_y_bounds(ax)
    panel_top = plot_top if plot_top is not None else CONTROL_PANEL_DEFAULT_TOP
    panel_height = max(panel_top - CONTROL_PANEL_MATURITY_BOTTOM, 0.01)
    toggle_ax = fig.add_axes(
        (
            CONTROL_PANEL_LEFT,
            CONTROL_PANEL_MATURITY_BOTTOM,
            CONTROL_PANEL_WIDTH,
            panel_height,
        )
    )
    toggle_ax.set_title("Maturity")
    toggle = CheckButtons(
        ax=toggle_ax,
        labels=maturities,
        actives=[maturity in visibility.visible_maturities for maturity in maturities],
    )
    colors = maturity_colors or _maturity_colors(plotted_series)
    for label, maturity in zip(toggle.labels, maturities):
        label.set_color(colors.get(maturity, "0.1"))

    def update(selected_maturity: str | None) -> None:
        if selected_maturity is None:
            return
        if selected_maturity in visibility.visible_maturities:
            visibility.visible_maturities.remove(selected_maturity)
        else:
            visibility.visible_maturities.add(selected_maturity)
        active_plotted_series = (
            plot_context.plotted_series if plot_context is not None else plotted_series
        )
        active_underlying_overlay = (
            plot_context.underlying_overlay if plot_context is not None else underlying_overlay
        )
        active_secondary_overlay = (
            plot_context.secondary_overlay if plot_context is not None else None
        )
        _refresh_visibility(
            fig,
            ax,
            active_plotted_series,
            visibility,
            active_underlying_overlay,
            active_secondary_overlay,
        )

    toggle.on_clicked(update)
    return toggle


def _set_plot_field(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
    field: str,
    title_prefix: str,
) -> None:
    field_spec = parse_plot_field(field)
    for series in plotted_series:
        series.line.set_ydata(series.quotes[field_spec.key])

    ax.relim(visible_only=True)
    ax.autoscale_view()
    _set_field_axis_direction(ax, field_spec.label)
    ax.set_ylabel(_field_ylabel(field_spec.label))
    _set_plot_title(ax, f"{title_prefix} · {field_spec.label}")
    _sync_maturity_guides(fig, ax, plotted_series)
    fig.canvas.draw_idle()


def add_field_toggle(
    fig: Figure,
    ax: Axes,
    plotted_series: list[PlotSeries],
    fields: tuple[str, ...],
    initial_field: str,
    title_prefix: str,
    *,
    plot_right: float = CONTROL_PANEL_PLOT_RIGHT,
    plot_context: PlotContext | None = None,
) -> RadioButtons | None:
    if len(fields) < 2:
        return None

    fig.subplots_adjust(right=plot_right)
    toggle_ax = fig.add_axes((CONTROL_PANEL_LEFT, 0.12, CONTROL_PANEL_WIDTH, 0.18))
    toggle = RadioButtons(
        ax=toggle_ax,
        labels=fields,
        active=fields.index(initial_field),
    )

    def update(selected_field: str | None) -> None:
        if selected_field is None:
            return
        if plot_context is not None:
            plot_context.field = selected_field
            _rebuild_plot_context(plot_context)
            return
        _set_plot_field(fig, ax, plotted_series, selected_field, title_prefix)

    toggle.on_clicked(update)
    return toggle


def _style_mode_button(button: Button, is_active: bool) -> None:
    button.color = MODE_TOGGLE_ACTIVE_COLOR if is_active else MODE_TOGGLE_INACTIVE_COLOR
    button.hovercolor = button.color
    button.ax.set_facecolor(button.color)
    button.label.set_color(
        MODE_TOGGLE_ACTIVE_TEXT_COLOR
        if is_active
        else MODE_TOGGLE_INACTIVE_TEXT_COLOR
    )


def _set_mode_toggle_state(toggle: ModeToggle, mode: str) -> None:
    _style_mode_button(toggle.latest_button, mode == LATEST_PLOT_MODE)
    _style_mode_button(toggle.historical_button, mode == HISTORICAL_PLOT_MODE)


def add_mode_toggle(
    fig: Figure,
    ax: Axes,
    visibility: PlotVisibility,
    *,
    plot_context: PlotContext | None = None,
) -> ModeToggle:
    plot_left, plot_right = _axis_box_x_bounds(ax)
    _, plot_top = _axis_box_y_bounds(ax)
    right = plot_right if plot_right is not None else 0.9
    left = max(
        right - ((MODE_TOGGLE_BUTTON_WIDTH * 2) + MODE_TOGGLE_BUTTON_GAP),
        plot_left if plot_left is not None else 0.1,
    )
    bottom = min(
        (plot_top if plot_top is not None else CONTROL_PANEL_DEFAULT_TOP)
        + MODE_TOGGLE_VERTICAL_OFFSET,
        0.98 - MODE_TOGGLE_BUTTON_HEIGHT,
    )
    latest_ax = fig.add_axes(
        (left, bottom, MODE_TOGGLE_BUTTON_WIDTH, MODE_TOGGLE_BUTTON_HEIGHT)
    )
    historical_ax = fig.add_axes(
        (
            left + MODE_TOGGLE_BUTTON_WIDTH + MODE_TOGGLE_BUTTON_GAP,
            bottom,
            MODE_TOGGLE_BUTTON_WIDTH,
            MODE_TOGGLE_BUTTON_HEIGHT,
        )
    )
    latest_button = Button(
        latest_ax,
        "Latest",
        color=MODE_TOGGLE_INACTIVE_COLOR,
        hovercolor=MODE_TOGGLE_INACTIVE_COLOR,
    )
    historical_button = Button(
        historical_ax,
        "Historical",
        color=MODE_TOGGLE_INACTIVE_COLOR,
        hovercolor=MODE_TOGGLE_INACTIVE_COLOR,
    )
    toggle = ModeToggle(
        latest_button=latest_button,
        historical_button=historical_button,
    )
    _set_mode_toggle_state(toggle, visibility.mode)

    def set_mode(mode: str) -> None:
        if visibility.mode == mode:
            return
        visibility.mode = mode
        _set_mode_toggle_state(toggle, mode)
        if plot_context is not None:
            _rebuild_plot_context(plot_context)

    latest_button.on_clicked(lambda _event: set_mode(LATEST_PLOT_MODE))
    historical_button.on_clicked(lambda _event: set_mode(HISTORICAL_PLOT_MODE))
    return toggle


def _remember_control(fig: Figure, name: str, control: object) -> None:
    setattr(fig, name, control)


def _set_plot_title(ax: Axes, title: str) -> None:
    ax.set_title(title, loc="left")


def _suppress_default_figure_title(fig: Figure) -> None:
    fig.set_label("")
    manager = getattr(fig.canvas, "manager", None)
    set_window_title = getattr(manager, "set_window_title", None)
    if callable(set_window_title):
        set_window_title("")


def _resolve_csv_path(arg: Path | None) -> Path:
    if arg is not None:
        return arg

    explicit = os.environ.get("PLOT_CSV")
    if explicit:
        return Path(explicit)

    symbol = os.environ.get("SYMBOL")
    side = os.environ.get("SIDE", "").lower()
    if not symbol or side not in ("put", "call"):
        raise SystemExit(
            "error: no CSV path given and PLOT_CSV / (SYMBOL + SIDE) not set in environment"
        )
    out_dir = Path(os.environ.get("OUT_DIR", "./data"))
    return out_dir / symbol.upper() / side / f"all_{side}s.csv"


def _resolve_field() -> str:
    return parse_plot_field(os.environ.get("PLOT_FIELD", "mid")).label


def _field_ylabel(field: str) -> str:
    if parse_plot_field(field).key == LOSS_FIELD:
        return "loss %"
    return f"{field} price"


def _set_field_axis_direction(ax: Axes, field: str) -> None:
    ax.yaxis.set_inverted(parse_plot_field(field).key == LOSS_FIELD)


def _remove_overlay_axes(plot_context: PlotContext) -> None:
    if plot_context.underlying_overlay is not None:
        plot_context.underlying_overlay.ax.remove()
        plot_context.underlying_overlay = None
    if plot_context.secondary_overlay is not None:
        plot_context.secondary_overlay.ax.remove()
        plot_context.secondary_overlay = None


def _rebuild_plot_context(plot_context: PlotContext) -> None:
    _remove_overlay_axes(plot_context)
    _clear_strike_end_labels(plot_context.fig)
    plot_context.ax.clear()

    plot_quotes = _plot_quotes_for_mode(plot_context)
    field_spec = parse_plot_field(plot_context.field)
    plot_context.plotted_series = build_plot_series(
        plot_context.ax,
        plot_quotes,
        field_spec.key,
        mode=plot_context.visibility.mode,
        maturity_colors=plot_context.maturity_colors,
    )
    if plot_context.visibility.mode == HISTORICAL_PLOT_MODE:
        plot_context.underlying_overlay = add_underlying_overlay(
            plot_context.ax,
            plot_context.plotted_series,
        )
        plot_context.secondary_overlay = None
    else:
        plot_context.secondary_overlay = add_latest_field_overlay(
            plot_context.ax,
            plot_quotes,
            _latest_mode_companion_field(plot_context.field),
            plot_context.maturity_colors,
        )
        plot_context.underlying_overlay = None

    _remember_control(plot_context.fig, "_underlying_overlay", plot_context.underlying_overlay)
    _remember_control(plot_context.fig, "_secondary_overlay", plot_context.secondary_overlay)
    _set_plot_title(plot_context.ax, f"{plot_context.title_prefix} · {field_spec.label}")
    plot_context.ax.set_xlabel(_plot_mode_xlabel(plot_context.visibility.mode))
    _set_field_axis_direction(plot_context.ax, field_spec.label)
    plot_context.ax.set_ylabel(_field_ylabel(field_spec.label))
    plot_context.ax.grid(True, alpha=0.3)
    if plot_context.visibility.mode == HISTORICAL_PLOT_MODE:
        plot_context.fig.autofmt_xdate()
    _refresh_visibility(
        plot_context.fig,
        plot_context.ax,
        plot_context.plotted_series,
        plot_context.visibility,
        plot_context.underlying_overlay,
        plot_context.secondary_overlay,
    )


def _resolve_initial_maturity(quotes: pd.DataFrame, requested: str | None) -> str:
    maturity = requested
    if maturity is None:
        maturity = latest_maturity(quotes)
    if maturity is None:
        raise SystemExit("no rows to plot")

    maturities = available_maturities(quotes)
    if maturity not in maturities:
        if len({float(cast(Any, strike)) for strike in quotes["strikePrice"].dropna().unique()}) == 1:
            strike_value = float(cast(Any, quotes["strikePrice"].dropna().iloc[0]))
            raise SystemExit(f"no rows for strike={strike_value} maturity={maturity}")
        raise SystemExit(f"no rows for maturity={maturity}")
    return maturity


def _title_prefix(quotes: pd.DataFrame, maturity: str, strike: float | None) -> str:
    maturities = available_maturities(quotes)
    strikes = sorted({float(cast(Any, value)) for value in quotes["strikePrice"].dropna().unique()})
    if strike is None:
        base = f"All strikes · Expiry {maturity}" if len(maturities) == 1 else "All strikes"
    elif len(maturities) == 1:
        base = f"Strike {strike} · Expiry {maturity}"
    elif len(strikes) == 1:
        base = f"Strike {strike}"
    else:
        base = "All strikes"

    underlying_title = str(quotes["underlying"].iloc[0])
    latest_underlying_price = _last_underlying_price(quotes)
    if latest_underlying_price is not None:
        underlying_title = (
            f"{underlying_title} @ {_format_display_value(latest_underlying_price)}"
        )

    return f"{underlying_title} · {base}"


def main() -> int:
    if load_dotenv is not None:
        load_dotenv(dotenv_path=Path(".env"))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=None,
        help="Path to combined option CSV (falls back to PLOT_CSV or OUT_DIR/SYMBOL/SIDE/all_{SIDE}s.csv from .env)",
    )
    parser.add_argument("--strike", type=float, default=None, help="Strike price to plot")
    parser.add_argument(
        "--maturity",
        default=None,
        help="Maturity date YYYY-MM-DD (default: latest maturity as the initial selection for the chosen strike, or latest overall without --strike)",
    )
    args = parser.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"error: CSV not found: {csv_path}")

    field = _resolve_field()
    field_spec = parse_plot_field(field)

    if field_spec.key == LOSS_FIELD:
        quotes = load_prepared_quotes(
            csv_path,
            crash_price_fraction=field_spec.crash_price_fraction,
            overwrite_loss=True,
            field_label=field_spec.label,
        )
    else:
        quotes = load_prepared_quotes(csv_path)
    _require_plot_field(quotes, field_spec.label)
    field_options = available_plot_fields(
        quotes,
        loss_field_label=(field_spec.label if field_spec.key == LOSS_FIELD else LOSS_FIELD),
    )
    filtered_quotes = filter_by_strike(quotes, args.strike)
    if filtered_quotes.empty:
        if args.strike is None:
            raise SystemExit(f"no rows in {csv_path}")
        raise SystemExit(f"no rows for strike={args.strike} in {csv_path}")

    maturity = _resolve_initial_maturity(filtered_quotes, args.maturity)
    if args.maturity is None:
        if args.strike is None:
            print(f"using latest maturity {maturity} as initial selection for all strikes")
        else:
            print(
                f"using latest maturity {maturity} as initial selection for strike {args.strike}"
            )

    fig, ax = plt.subplots(figsize=(10, 5))
    _suppress_default_figure_title(fig)
    slider = None
    maturity_toggle = None
    field_toggle = None
    mode_toggle = None
    maturities = available_maturities(filtered_quotes)
    maturity_colors = _build_maturity_color_map(maturities)
    visibility = PlotVisibility(
        visible_maturities={maturity},
        mode=LATEST_PLOT_MODE,
    )
    if args.strike is None:
        visibility.strike_bounds = _default_strike_range_bounds(filtered_quotes)
    title_prefix = _title_prefix(filtered_quotes, maturity, args.strike)
    control_panel_plot_right = _plot_right_for_quotes(filtered_quotes, field)
    plot_context = PlotContext(
        fig=fig,
        ax=ax,
        quotes=filtered_quotes,
        latest_quotes=filter_to_latest_timestamp(filtered_quotes),
        title_prefix=title_prefix,
        field=field,
        visibility=visibility,
        maturities=maturities,
        maturity_colors=maturity_colors,
        plot_right=control_panel_plot_right,
        plotted_series=[],
    )
    _rebuild_plot_context(plot_context)
    fig.tight_layout()
    fig.subplots_adjust(right=control_panel_plot_right)
    if visibility is not None:
        if filtered_quotes["strikePrice"].dropna().nunique() > 1:
            slider = add_strike_range_slider(
                fig,
                ax,
                plot_context.plotted_series,
                visibility,
                plot_context.underlying_overlay,
                reference_price=_last_underlying_price(filtered_quotes),
                plot_right=control_panel_plot_right,
                plot_context=plot_context,
            )
            if slider is not None:
                _remember_control(fig, "_strike_range_slider", slider)
        maturity_toggle = add_maturity_toggle(
            fig,
            ax,
            plot_context.plotted_series,
            maturities,
            visibility,
            plot_context.underlying_overlay,
            plot_right=control_panel_plot_right,
            maturity_colors=maturity_colors,
            plot_context=plot_context,
        )
        if maturity_toggle is not None:
            _remember_control(fig, "_maturity_toggle", maturity_toggle)
    field_toggle = add_field_toggle(
        fig,
        ax,
        plot_context.plotted_series,
        field_options,
        field,
        title_prefix,
        plot_right=control_panel_plot_right,
        plot_context=plot_context,
    )
    if field_toggle is not None:
        _remember_control(fig, "_field_toggle", field_toggle)
    mode_toggle = add_mode_toggle(
        fig,
        ax,
        visibility,
        plot_context=plot_context,
    )
    _remember_control(fig, "_mode_toggle", mode_toggle)
    _refresh_visibility(
        fig,
        ax,
        plot_context.plotted_series,
        visibility,
        plot_context.underlying_overlay,
        plot_context.secondary_overlay,
    )
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
