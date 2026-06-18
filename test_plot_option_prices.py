import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd

import plot_option_prices as module


class ResolveFieldTests(unittest.TestCase):
    def test_resolve_field_accepts_loss(self) -> None:
        with patch.dict(module.os.environ, {"PLOT_FIELD": "loss"}, clear=True):
            self.assertEqual(module._resolve_field(), "loss")

    def test_resolve_field_accepts_loss_with_custom_crash_fraction(self) -> None:
        with patch.dict(module.os.environ, {"PLOT_FIELD": "loss@30%"}, clear=True):
            self.assertEqual(module._resolve_field(), "loss@30%")

    def test_resolve_field_rejects_invalid_custom_loss_fraction(self) -> None:
        with patch.dict(module.os.environ, {"PLOT_FIELD": "loss@0%"}, clear=True):
            with self.assertRaises(SystemExit) as exc_info:
                module._resolve_field()

        self.assertEqual(
            str(exc_info.exception),
            "error: PLOT_FIELD loss percent must be greater than 0 and at most 100",
        )


class EnrichQuotesTests(unittest.TestCase):
    def test_enrich_quotes_adds_loss_and_exposes_it_as_available(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621P00090000", "AAPL240621P00015000"],
                "underlyingPrice": [100.0, 100.0],
                "strikePrice": [90.0, 15.0],
                "ask": [2.0, 1.0],
            }
        )

        prepared = module.enrich_quotes(quotes)

        self.assertAlmostEqual(cast(float, prepared.loc[0, "loss"]), 12.0)
        self.assertAlmostEqual(cast(float, prepared.loc[1, "loss"]), 81.0)
        self.assertEqual(module.available_plot_fields(prepared), ("ask", "loss"))

    def test_enrich_quotes_skips_loss_when_input_does_not_support_it(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621C00100000"],
                "underlyingPrice": [100.0],
                "strikePrice": [100.0],
                "ask": [2.0],
                "mid": [1.5],
            }
        )

        prepared = module.enrich_quotes(quotes)

        self.assertNotIn("loss", prepared.columns)
        self.assertEqual(module.available_plot_fields(prepared), ("mid", "ask"))


class PrepareQuotesForFieldTests(unittest.TestCase):
    def test_prepare_quotes_for_field_computes_protective_put_loss_pct(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621P00090000", "AAPL240621P00015000"],
                "underlyingPrice": [100.0, 100.0],
                "strikePrice": [90.0, 15.0],
                "ask": [2.0, 1.0],
            }
        )

        prepared = module.prepare_quotes_for_field(quotes, "loss")

        self.assertAlmostEqual(cast(float, prepared.loc[0, "loss"]), 12.0)
        self.assertAlmostEqual(cast(float, prepared.loc[1, "loss"]), 81.0)

    def test_prepare_quotes_for_field_supports_custom_loss_crash_fraction(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621P00090000", "AAPL240621P00015000"],
                "underlyingPrice": [100.0, 100.0],
                "strikePrice": [90.0, 15.0],
                "ask": [2.0, 1.0],
            }
        )

        prepared = module.prepare_quotes_for_field(quotes, "loss@30%")

        self.assertAlmostEqual(cast(float, prepared.loc[0, "loss"]), 12.0)
        self.assertAlmostEqual(cast(float, prepared.loc[1, "loss"]), 71.0)

    def test_prepare_quotes_for_field_rejects_call_data(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621C00100000"],
                "underlyingPrice": [100.0],
                "strikePrice": [100.0],
                "ask": [2.0],
            }
        )

        with self.assertRaises(SystemExit) as exc_info:
            module.prepare_quotes_for_field(quotes, "loss")

        self.assertEqual(
            str(exc_info.exception),
            "error: PLOT_FIELD='loss' is only supported for put option data",
        )

    def test_prepare_quotes_for_field_rejects_call_data_for_custom_loss(self) -> None:
        quotes = pd.DataFrame(
            {
                "optionSymbol": ["AAPL240621C00100000"],
                "underlyingPrice": [100.0],
                "strikePrice": [100.0],
                "ask": [2.0],
            }
        )

        with self.assertRaises(SystemExit) as exc_info:
            module.prepare_quotes_for_field(quotes, "loss@30%")

        self.assertEqual(
            str(exc_info.exception),
            "error: PLOT_FIELD='loss@30%' is only supported for put option data",
        )


class PlotSingleContractTests(unittest.TestCase):
    def test_plot_single_contract_adds_small_marker_by_default(self) -> None:
        quotes = pd.DataFrame(
            {
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "mid": [1.0, 1.1],
            }
        )
        fake_ax = MagicMock()

        module.plot_single_contract(fake_ax, quotes, "mid", label="Strike 100")

        _, kwargs = fake_ax.plot.call_args
        self.assertEqual(kwargs["label"], "Strike 100")
        self.assertEqual(kwargs["marker"], "o")
        self.assertEqual(kwargs["markersize"], 3)


class DefaultStrikeRangeTests(unittest.TestCase):
    def test_default_strike_range_bounds_targets_twenty_to_thirty_pct_below_latest_underlying(
        self,
    ) -> None:
        quotes = pd.DataFrame(
            {
                "updated": pd.to_datetime(
                    [
                        "2024-03-01",
                        "2024-03-02",
                        "2024-03-02",
                        "2024-03-02",
                    ]
                ),
                "strikePrice": [70.0, 75.0, 80.0, 85.0],
                "underlyingPrice": [99.0, 100.0, 100.0, 100.0],
            }
        )

        with patch.dict(module.os.environ, {}, clear=True):
            self.assertEqual(module._default_strike_range_bounds(quotes), (70.0, 80.0))

    def test_default_strike_range_bounds_uses_env_fraction_overrides(self) -> None:
        quotes = pd.DataFrame(
            {
                "updated": pd.to_datetime(
                    [
                        "2024-03-02",
                        "2024-03-02",
                        "2024-03-02",
                        "2024-03-02",
                        "2024-03-02",
                    ]
                ),
                "strikePrice": [75.0, 80.0, 85.0, 90.0, 95.0],
                "underlyingPrice": [100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )

        with patch.dict(
            module.os.environ,
            {
                "INITIAL_STRIKE_MIN_PRICE_FRACTION": "0.8",
                "INITIAL_STRIKE_MAX_PRICE_FRACTION": "0.95",
            },
            clear=True,
        ):
            self.assertEqual(module._default_strike_range_bounds(quotes), (80.0, 95.0))


class StrikeRangeSliderTests(unittest.TestCase):
    def test_align_slider_label_with_y_axis_labels_uses_leftmost_visible_tick_label(
        self,
    ) -> None:
        class FakeBBox:
            def __init__(self, x0: float) -> None:
                self.x0 = x0

            def transformed(self, _transform: object) -> "FakeBBox":
                return self

        fake_renderer = object()
        fake_fig = MagicMock()
        fake_fig.transFigure.inverted.return_value = object()
        fake_fig.canvas.get_renderer.return_value = fake_renderer
        fake_plot_ax = MagicMock()
        left_tick = MagicMock()
        left_tick.get_visible.return_value = True
        left_tick.get_text.return_value = "20"
        left_tick.get_window_extent.return_value = FakeBBox(0.03)
        right_tick = MagicMock()
        right_tick.get_visible.return_value = True
        right_tick.get_text.return_value = "30"
        right_tick.get_window_extent.return_value = FakeBBox(0.04)
        hidden_tick = MagicMock()
        hidden_tick.get_visible.return_value = False
        hidden_tick.get_text.return_value = "10"
        hidden_tick.get_window_extent.return_value = FakeBBox(0.01)
        fake_plot_ax.get_yticklabels.return_value = [right_tick, hidden_tick, left_tick]
        fake_plot_ax.get_position.return_value.x1 = 0.82
        fake_slider_ax = MagicMock()
        fake_slider_ax.get_position.return_value.bounds = (0.18, 0.04, 0.64, 0.04)
        fake_slider = MagicMock()
        fake_slider.label.get_window_extent.return_value = FakeBBox(0.06)

        module._align_slider_label_with_y_axis_labels(
            fake_fig,
            fake_plot_ax,
            fake_slider_ax,
            fake_slider,
        )

        fake_slider_ax.set_position.assert_called_once_with((0.15, 0.04, 0.67, 0.04))

    def test_format_strike_range_text_uses_brackets_and_percentage_range(self) -> None:
        self.assertEqual(
            module._format_strike_range_text((100.0, 105.0), 125.0),
            "[100, 105] [80%-84%]",
        )

    def test_format_strike_range_text_rounds_percentages_to_integers(self) -> None:
        self.assertEqual(
            module._format_strike_range_text((100.0, 104.0), 123.0),
            "[100, 104] [81%-85%]",
        )

    def test_add_strike_range_slider_filters_visible_lines(self) -> None:
        fake_fig = MagicMock()
        fake_fig.add_axes.return_value = MagicMock()
        fake_ax = MagicMock()
        fake_ax.legend_ = None
        line_100 = MagicMock()
        line_105 = MagicMock()
        line_100.get_color.return_value = "C0"
        line_105.get_color.return_value = "C1"
        fake_slider = MagicMock()
        plotted_series = [
            module.PlotSeries(line_100, pd.DataFrame(), "2024-03-15", 100.0),
            module.PlotSeries(line_105, pd.DataFrame(), "2024-06-21", 105.0),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15", "2024-06-21"}
        )

        with patch.object(module, "RangeSlider", return_value=fake_slider) as slider_cls:
            slider = module.add_strike_range_slider(
                fake_fig,
                fake_ax,
                plotted_series,
                visibility,
            )

        self.assertIs(slider, fake_slider)
        self.assertEqual(visibility.strike_bounds, (100.0, 105.0))
        fake_fig.subplots_adjust.assert_called_once_with(
            bottom=module.STRIKE_SLIDER_PLOT_BOTTOM
        )
        fake_fig.add_axes.assert_called_once_with(
            (
                module.STRIKE_SLIDER_LEFT,
                module.STRIKE_SLIDER_BOTTOM,
                module.STRIKE_SLIDER_WIDTH,
                module.STRIKE_SLIDER_HEIGHT,
            )
        )
        slider_cls.assert_called_once_with(
            ax=fake_fig.add_axes.return_value,
            label=module.STRIKE_SLIDER_LABEL,
            valmin=100.0,
            valmax=105.0,
            valinit=(100.0, 105.0),
            valstep=[100.0, 105.0],
            valfmt="%0.0f",
        )
        self.assertEqual(
            [call.args[0] for call in fake_slider.valtext.set_text.call_args_list],
            ["[100, 105]"],
        )

        callback = fake_slider.on_changed.call_args.args[0]
        callback((105.0, 105.0))

        line_100.set_visible.assert_called_once_with(False)
        line_105.set_visible.assert_called_once_with(True)
        self.assertEqual(visibility.strike_bounds, (105.0, 105.0))
        self.assertEqual(
            [call.args[0] for call in fake_slider.valtext.set_text.call_args_list],
            ["[100, 105]", "[105, 105]"],
        )
        fake_ax.relim.assert_called_once_with(visible_only=True)
        fake_ax.autoscale_view.assert_called_once_with()
        fake_fig.canvas.draw_idle.assert_called_once_with()

    def test_add_strike_range_slider_shows_percentage_range_from_reference_price(
        self,
    ) -> None:
        fake_fig = MagicMock()
        fake_fig.add_axes.return_value = MagicMock()
        fake_ax = MagicMock()
        fake_slider = MagicMock()
        plotted_series = [
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 100.0),
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-06-21", 105.0),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15", "2024-06-21"}
        )

        with patch.object(module, "RangeSlider", return_value=fake_slider):
            module.add_strike_range_slider(
                fake_fig,
                fake_ax,
                plotted_series,
                visibility,
                reference_price=125.0,
            )

        fake_slider.valtext.set_text.assert_called_once_with(
            "[100, 105] [80%-84%]"
        )

    def test_add_strike_range_slider_uses_existing_visibility_bounds_for_initial_range(
        self,
    ) -> None:
        fake_fig = MagicMock()
        fake_fig.add_axes.return_value = MagicMock()
        fake_ax = MagicMock()
        fake_slider = MagicMock()
        plotted_series = [
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 80.0),
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 85.0),
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 90.0),
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 95.0),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15"},
            strike_bounds=(85.0, 90.0),
        )

        with patch.object(module, "RangeSlider", return_value=fake_slider) as slider_cls:
            slider = module.add_strike_range_slider(
                fake_fig,
                fake_ax,
                plotted_series,
                visibility,
            )

        self.assertIs(slider, fake_slider)
        self.assertEqual(visibility.strike_bounds, (85.0, 90.0))
        self.assertEqual(slider_cls.call_args.kwargs["valinit"], (85.0, 90.0))


class XTickLabelAlignmentTests(unittest.TestCase):
    def test_ensure_leftmost_xtick_visible_expands_left_margin_when_needed(self) -> None:
        class FakeBBox:
            def __init__(self, x0: float) -> None:
                self.x0 = x0

            def transformed(self, _transform: object) -> "FakeBBox":
                return self

        first_label = MagicMock()
        first_label.get_visible.return_value = True
        first_label.get_text.return_value = "2024-01-01"
        first_label.get_window_extent.return_value = FakeBBox(0.005)
        middle_label = MagicMock()
        middle_label.get_visible.return_value = True
        middle_label.get_text.return_value = "2024-01-02"
        middle_label.get_window_extent.return_value = FakeBBox(0.12)
        last_label = MagicMock()
        last_label.get_visible.return_value = True
        last_label.get_text.return_value = "2024-01-03"
        last_label.get_window_extent.return_value = FakeBBox(0.25)
        hidden_label = MagicMock()
        hidden_label.get_visible.return_value = False
        hidden_label.get_text.return_value = "2023-12-31"
        fake_fig = MagicMock()
        fake_fig.transFigure.inverted.return_value = object()
        fake_ax = MagicMock()
        fake_ax.get_xticklabels.return_value = [hidden_label, first_label, middle_label, last_label]
        fake_ax.get_position.return_value.x0 = 0.05
        fake_ax.get_position.return_value.x1 = 0.8

        module._ensure_leftmost_xtick_visible(fake_fig, fake_ax)

        for label in (first_label, middle_label, last_label):
            label.set_rotation.assert_called_once_with(module.XTICK_LABEL_ROTATION)
            label.set_rotation_mode.assert_called_once_with("anchor")
            label.set_ha.assert_called_once_with("right")
        hidden_label.set_rotation_mode.assert_not_called()
        hidden_label.set_rotation.assert_not_called()
        hidden_label.set_ha.assert_not_called()
        fake_ax.tick_params.assert_called_once_with(
            axis="x",
            labelrotation=module.XTICK_LABEL_ROTATION,
        )
        fake_fig.subplots_adjust.assert_called_once_with(left=0.055)

    def test_refresh_visibility_keeps_leftmost_xtick_visible(self) -> None:
        fake_fig = MagicMock()
        fake_ax = MagicMock()
        line = MagicMock()
        line.get_visible.return_value = True
        plotted_series = [
            module.PlotSeries(line, pd.DataFrame(), "2024-03-15", 100.0)
        ]
        visibility = module.PlotVisibility(visible_maturities={"2024-03-15"})

        with patch.object(module, "_ensure_leftmost_xtick_visible") as align_mock:
            module._refresh_visibility(fake_fig, fake_ax, plotted_series, visibility)

        align_mock.assert_called_once_with(fake_fig, fake_ax)


class MaturityToggleTests(unittest.TestCase):
    def test_add_maturity_toggle_filters_visible_maturities(self) -> None:
        fake_fig = MagicMock()
        fake_toggle_ax = MagicMock()
        fake_fig.add_axes.return_value = fake_toggle_ax
        fake_ax = MagicMock()
        fake_ax.get_position.return_value.y1 = 0.88
        fake_ax.legend_ = None
        line_mar = MagicMock()
        line_jun = MagicMock()
        line_mar.get_color.return_value = "C0"
        line_jun.get_color.return_value = "C1"
        fake_toggle = MagicMock()
        fake_toggle.labels = [MagicMock(), MagicMock()]
        plotted_series = [
            module.PlotSeries(line_mar, pd.DataFrame(), "2024-03-15", 100.0),
            module.PlotSeries(line_jun, pd.DataFrame(), "2024-06-21", 100.0),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15", "2024-06-21"}
        )

        with patch.object(module, "CheckButtons", return_value=fake_toggle) as toggle_cls:
            toggle = module.add_maturity_toggle(
                fake_fig,
                fake_ax,
                plotted_series,
                ("2024-03-15", "2024-06-21"),
                visibility,
            )

        self.assertIs(toggle, fake_toggle)
        fake_fig.subplots_adjust.assert_called_once_with(
            right=module.CONTROL_PANEL_PLOT_RIGHT
        )
        fake_fig.add_axes.assert_called_once_with(
            (
                module.CONTROL_PANEL_LEFT,
                module.CONTROL_PANEL_MATURITY_BOTTOM,
                module.CONTROL_PANEL_WIDTH,
                0.88 - module.CONTROL_PANEL_MATURITY_BOTTOM,
            )
        )
        fake_toggle_ax.set_title.assert_called_once_with("Maturity")
        toggle_cls.assert_called_once_with(
            ax=fake_toggle_ax,
            labels=("2024-03-15", "2024-06-21"),
            actives=[True, True],
        )
        fake_toggle.labels[0].set_color.assert_called_once_with("C0")
        fake_toggle.labels[1].set_color.assert_called_once_with("C1")

        callback = fake_toggle.on_clicked.call_args.args[0]
        callback("2024-03-15")

        self.assertEqual(visibility.visible_maturities, {"2024-06-21"})
        line_mar.set_visible.assert_called_once_with(False)
        line_jun.set_visible.assert_called_once_with(True)
        fake_ax.relim.assert_called_once_with(visible_only=True)
        fake_ax.autoscale_view.assert_called_once_with()
        fake_fig.canvas.draw_idle.assert_called_once_with()

    def test_add_maturity_toggle_uses_overlay_plot_right_when_requested(self) -> None:
        fake_fig = MagicMock()
        fake_toggle_ax = MagicMock()
        fake_fig.add_axes.return_value = fake_toggle_ax
        fake_ax = MagicMock()
        fake_toggle = MagicMock()
        fake_toggle.labels = [MagicMock(), MagicMock()]
        plotted_series = [
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-03-15", 100.0),
            module.PlotSeries(MagicMock(), pd.DataFrame(), "2024-06-21", 100.0),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15", "2024-06-21"}
        )

        with patch.object(module, "CheckButtons", return_value=fake_toggle):
            module.add_maturity_toggle(
                fake_fig,
                fake_ax,
                plotted_series,
                ("2024-03-15", "2024-06-21"),
                visibility,
                plot_right=module.CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY,
            )

        fake_fig.subplots_adjust.assert_called_once_with(
            right=module.CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY
        )


class MaturityGuidesTests(unittest.TestCase):
    def test_refresh_visibility_adds_maturity_legend_and_strike_end_labels(self) -> None:
        fake_fig = MagicMock()
        fake_ax = MagicMock()
        fake_ax.legend_ = None
        annotation_one = MagicMock()
        annotation_two = MagicMock()
        fake_ax.annotate.side_effect = [annotation_one, annotation_two]
        plotted_series = [
            module.PlotSeries(
                module.Line2D(
                    pd.to_datetime(["2024-03-01", "2024-03-02"]),
                    [1.0, 1.1],
                    color="C0",
                ),
                pd.DataFrame(),
                "2024-03-15",
                100.0,
            ),
            module.PlotSeries(
                module.Line2D(
                    pd.to_datetime(["2024-03-01", "2024-03-02"]),
                    [2.0, 2.1],
                    color="C1",
                ),
                pd.DataFrame(),
                "2024-06-21",
                105.0,
            ),
        ]
        visibility = module.PlotVisibility(
            visible_maturities={"2024-03-15", "2024-06-21"}
        )

        module._refresh_visibility(fake_fig, fake_ax, plotted_series, visibility)

        fake_ax.set_xmargin.assert_called_once_with(module.END_LABEL_X_MARGIN)
        handles = fake_ax.legend.call_args.kwargs["handles"]
        self.assertEqual(
            [handle.get_label() for handle in handles],
            ["2024-03-15", "2024-06-21"],
        )
        self.assertEqual(
            [handle.get_color() for handle in handles],
            ["C0", "C1"],
        )
        self.assertEqual(fake_ax.legend.call_args.kwargs["title"], "Maturity")
        self.assertEqual(fake_ax.annotate.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in fake_ax.annotate.call_args_list],
            ["100", "105"],
        )
        self.assertEqual(
            [call.kwargs["color"] for call in fake_ax.annotate.call_args_list],
            ["C0", "C1"],
        )
        self.assertEqual(
            fake_fig._strike_end_labels,
            [annotation_one, annotation_two],
        )
        fake_fig.canvas.draw_idle.assert_called_once_with()


class UnderlyingOverlayTests(unittest.TestCase):
    def test_add_underlying_overlay_uses_solid_thicker_line(self) -> None:
        fake_ax = MagicMock()
        fake_overlay_ax = MagicMock()
        fake_ax.twinx.return_value = fake_overlay_ax
        fake_line = MagicMock()
        fake_overlay_ax.plot.return_value = [fake_line]
        visible_line = MagicMock()
        visible_line.get_visible.return_value = True

        overlay = module.add_underlying_overlay(
            fake_ax,
            [
                module.PlotSeries(
                    visible_line,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01"]),
                            "underlyingPrice": [100.0],
                        }
                    ),
                    "2024-03-15",
                    100.0,
                )
            ],
        )

        self.assertIsNotNone(overlay)
        fake_overlay_ax.plot.assert_called_once_with(
            [],
            [],
            color="0.25",
            linestyle="-",
            linewidth=2.5,
            alpha=0.85,
        )

    def test_visible_underlying_price_summary_aggregates_visible_rows(self) -> None:
        visible_line = MagicMock()
        visible_line.get_visible.return_value = True
        hidden_line = MagicMock()
        hidden_line.get_visible.return_value = False

        summary = module._visible_underlying_price_summary(
            [
                module.PlotSeries(
                    visible_line,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                            "underlyingPrice": [100.0, 101.0],
                        }
                    ),
                    "2024-03-15",
                    100.0,
                ),
                module.PlotSeries(
                    visible_line,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                            "underlyingPrice": [100.5, 101.0],
                        }
                    ),
                    "2024-06-21",
                    105.0,
                ),
                module.PlotSeries(
                    hidden_line,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01"]),
                            "underlyingPrice": [999.0],
                        }
                    ),
                    "2024-09-20",
                    110.0,
                ),
            ]
        )

        self.assertEqual(
            summary["updated"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-03-01", "2024-03-02"],
        )
        self.assertAlmostEqual(cast(float, summary.loc[0, "underlyingPriceMean"]), 100.25)
        self.assertAlmostEqual(cast(float, summary.loc[0, "underlyingPriceMin"]), 100.0)
        self.assertAlmostEqual(cast(float, summary.loc[0, "underlyingPriceMax"]), 100.5)
        self.assertAlmostEqual(cast(float, summary.loc[1, "underlyingPriceMean"]), 101.0)
        self.assertAlmostEqual(cast(float, summary.loc[1, "underlyingPriceMin"]), 101.0)
        self.assertAlmostEqual(cast(float, summary.loc[1, "underlyingPriceMax"]), 101.0)

    def test_visible_underlying_price_summary_falls_back_when_all_maturities_hidden(self) -> None:
        hidden_line_one = MagicMock()
        hidden_line_one.get_visible.return_value = False
        hidden_line_two = MagicMock()
        hidden_line_two.get_visible.return_value = False

        summary = module._visible_underlying_price_summary(
            [
                module.PlotSeries(
                    hidden_line_one,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                            "underlyingPrice": [100.0, 101.0],
                        }
                    ),
                    "2024-03-15",
                    100.0,
                ),
                module.PlotSeries(
                    hidden_line_two,
                    pd.DataFrame(
                        {
                            "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                            "underlyingPrice": [999.0, 999.0],
                        }
                    ),
                    "2024-06-21",
                    105.0,
                ),
            ],
            module.PlotVisibility(visible_maturities=set(), strike_bounds=(100.0, 100.0)),
        )

        self.assertEqual(
            summary["updated"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-03-01", "2024-03-02"],
        )
        self.assertEqual(summary["underlyingPriceMean"].tolist(), [100.0, 101.0])

    def test_refresh_visibility_updates_underlying_overlay(self) -> None:
        fake_fig = MagicMock()
        fake_ax = MagicMock()
        line = MagicMock()
        overlay = module.UnderlyingOverlay(ax=MagicMock(), line=MagicMock())
        plotted_series = [
            module.PlotSeries(line, pd.DataFrame(), "2024-03-15", 100.0)
        ]
        visibility = module.PlotVisibility(visible_maturities={"2024-03-15"})

        with patch.object(module, "_update_underlying_overlay") as overlay_mock:
            module._refresh_visibility(
                fake_fig,
                fake_ax,
                plotted_series,
                visibility,
                overlay,
            )

        overlay_mock.assert_called_once_with(overlay, plotted_series, visibility)


class FieldToggleTests(unittest.TestCase):
    def test_add_field_toggle_switches_line_data_and_labels(self) -> None:
        fake_fig = MagicMock()
        fake_toggle_ax = MagicMock()
        fake_fig.add_axes.return_value = fake_toggle_ax
        fake_ax = MagicMock()
        fake_ax.legend_ = None
        fake_line = MagicMock()
        fake_toggle = MagicMock()
        quotes = pd.DataFrame(
            {
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "mid": [1.0, 1.1],
                "loss": [12.0, 13.0],
            }
        )

        with patch.object(module, "RadioButtons", return_value=fake_toggle) as toggle_cls:
            toggle = module.add_field_toggle(
                fake_fig,
                fake_ax,
                [
                    module.PlotSeries(
                        fake_line,
                        quotes,
                        "2024-06-21",
                        90.0,
                    )
                ],
                ("mid", "loss"),
                "mid",
                "Strike 90 · Expiry 2024-06-21",
            )

        self.assertIs(toggle, fake_toggle)
        fake_fig.subplots_adjust.assert_called_once_with(
            right=module.CONTROL_PANEL_PLOT_RIGHT
        )
        fake_fig.add_axes.assert_called_once_with(
            (module.CONTROL_PANEL_LEFT, 0.12, module.CONTROL_PANEL_WIDTH, 0.18)
        )
        fake_toggle_ax.set_title.assert_not_called()
        toggle_cls.assert_called_once_with(
            ax=fake_toggle_ax,
            labels=("mid", "loss"),
            active=0,
        )

        callback = fake_toggle.on_clicked.call_args.args[0]
        callback("loss")

        fake_line.set_ydata.assert_called_once_with(quotes["loss"])
        fake_ax.relim.assert_called_once_with(visible_only=True)
        fake_ax.autoscale_view.assert_called_once_with()
        fake_ax.yaxis.set_inverted.assert_called_once_with(True)
        fake_ax.set_ylabel.assert_called_once_with("loss %")
        fake_ax.set_title.assert_called_once_with("Strike 90 · Expiry 2024-06-21 · loss")
        fake_fig.canvas.draw_idle.assert_called_once_with()

    def test_add_field_toggle_uses_overlay_plot_right_when_requested(self) -> None:
        fake_fig = MagicMock()
        fake_toggle_ax = MagicMock()
        fake_fig.add_axes.return_value = fake_toggle_ax
        fake_ax = MagicMock()
        fake_toggle = MagicMock()

        with patch.object(module, "RadioButtons", return_value=fake_toggle):
            module.add_field_toggle(
                fake_fig,
                fake_ax,
                [
                    module.PlotSeries(
                        MagicMock(),
                        pd.DataFrame({"mid": [1.0], "updated": pd.to_datetime(["2024-03-01"])}),
                        "2024-06-21",
                        90.0,
                    )
                ],
                ("mid", "loss"),
                "mid",
                "Strike 90 · Expiry 2024-06-21",
                plot_right=module.CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY,
            )

        fake_fig.subplots_adjust.assert_called_once_with(
            right=module.CONTROL_PANEL_PLOT_RIGHT_WITH_OVERLAY
        )


class PlotAllStrikesTests(unittest.TestCase):
    def test_main_suppresses_default_figure_title(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL", "AAPL"],
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "strikePrice": [90.0, 90.0],
                "maturityDate": ["2024-06-21", "2024-06-21"],
                "mid": [1.0, 1.1],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path), "--strike", "90"]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "mid"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            rc = module.main()

        self.assertEqual(rc, 0)
        fake_fig.set_label.assert_called_once_with("")
        fake_fig.canvas.manager.set_window_title.assert_called_once_with("")

    def test_main_without_strike_plots_each_strike_for_all_maturities(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL"] * 5,
                "updated": pd.to_datetime(
                    [
                        "2024-03-01",
                        "2024-03-02",
                        "2024-03-01",
                        "2024-03-02",
                        "2024-02-20",
                    ]
                ),
                "strikePrice": [100.0, 100.0, 105.0, 105.0, 90.0],
                "maturityDate": [
                    "2024-06-21",
                    "2024-06-21",
                    "2024-06-21",
                    "2024-06-21",
                    "2024-03-15",
                ],
                "mid": [1.0, 1.1, 2.0, 2.1, 0.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()
            fake_ax.legend_ = None
            slider_sentinel = object()
            maturity_toggle_sentinel = object()
            field_toggle_sentinel = object()
            plotted_lines = []

            for color in ("C0", "C1", "C1"):
                line = MagicMock()
                line.get_color.return_value = color
                plotted_lines.append(line)

            def add_slider(*args: object, **kwargs: object) -> object:
                fake_fig.tight_layout.assert_called_once_with()
                return slider_sentinel

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path)]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "mid"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            with patch.object(module, "plot_single_contract") as plot_mock:
                                plot_mock.side_effect = plotted_lines
                                with patch.object(
                                    module,
                                    "add_strike_range_slider",
                                    side_effect=add_slider,
                                ) as slider_mock:
                                    with patch.object(
                                        module,
                                        "add_maturity_toggle",
                                        return_value=maturity_toggle_sentinel,
                                    ) as maturity_mock:
                                        with patch.object(
                                            module,
                                            "add_field_toggle",
                                            return_value=field_toggle_sentinel,
                                        ) as field_mock:
                                            rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(plot_mock.call_count, 3)
        slider_mock.assert_called_once()
        maturity_mock.assert_called_once()
        field_mock.assert_called_once()
        fake_fig.tight_layout.assert_called_once_with()
        self.assertIs(fake_fig._strike_range_slider, slider_sentinel)
        self.assertIs(fake_fig._maturity_toggle, maturity_toggle_sentinel)
        self.assertIs(fake_fig._field_toggle, field_toggle_sentinel)

        plotted_strikes = [call.kwargs["label"] for call in plot_mock.call_args_list]
        self.assertEqual(
            plotted_strikes,
            [
                "Expiry 2024-03-15 · Strike 90",
                "Expiry 2024-06-21 · Strike 100",
                "Expiry 2024-06-21 · Strike 105",
            ],
        )

        self.assertEqual(
            [call.kwargs["color"] for call in plot_mock.call_args_list],
            ["C0", "C1", "C1"],
        )
        for call in plot_mock.call_args_list:
            self.assertEqual(call.kwargs["linestyle"], "-")

    def test_main_with_strike_plots_all_maturities_for_selected_strike(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL"] * 5,
                "updated": pd.to_datetime(
                    [
                        "2024-02-20",
                        "2024-02-21",
                        "2024-03-01",
                        "2024-03-02",
                        "2024-03-01",
                    ]
                ),
                "strikePrice": [90.0, 90.0, 90.0, 90.0, 95.0],
                "maturityDate": [
                    "2024-03-15",
                    "2024-03-15",
                    "2024-06-21",
                    "2024-06-21",
                    "2024-06-21",
                ],
                "mid": [0.5, 0.6, 1.0, 1.1, 2.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()
            fake_ax.legend_ = None
            maturity_toggle_sentinel = object()
            field_toggle_sentinel = object()
            plotted_lines = []

            for color in ("C0", "C1"):
                line = MagicMock()
                line.get_color.return_value = color
                plotted_lines.append(line)

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path), "--strike", "90"]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "mid"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            with patch.object(module, "plot_single_contract") as plot_mock:
                                plot_mock.side_effect = plotted_lines
                                with patch.object(
                                    module,
                                    "add_strike_range_slider",
                                ) as slider_mock:
                                    with patch.object(
                                        module,
                                        "add_maturity_toggle",
                                        return_value=maturity_toggle_sentinel,
                                    ) as maturity_mock:
                                        with patch.object(
                                            module,
                                            "add_field_toggle",
                                            return_value=field_toggle_sentinel,
                                        ) as field_mock:
                                            rc = module.main()

        self.assertEqual(rc, 0)
        self.assertEqual(plot_mock.call_count, 2)
        slider_mock.assert_not_called()
        maturity_mock.assert_called_once()
        field_mock.assert_called_once()
        self.assertIs(fake_fig._maturity_toggle, maturity_toggle_sentinel)
        self.assertIs(fake_fig._field_toggle, field_toggle_sentinel)

        plotted_labels = [call.kwargs["label"] for call in plot_mock.call_args_list]
        self.assertEqual(
            plotted_labels,
            ["Expiry 2024-03-15", "Expiry 2024-06-21"],
        )
        self.assertEqual(
            [call.kwargs["color"] for call in plot_mock.call_args_list],
            ["C0", "C1"],
        )
        for call in plot_mock.call_args_list:
            self.assertEqual(call.kwargs["linestyle"], "-")

    def test_main_with_loss_field_sets_percent_ylabel(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL", "AAPL"],
                "optionSymbol": ["AAPL240621P00090000", "AAPL240621P00090000"],
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "strikePrice": [90.0, 90.0],
                "maturityDate": ["2024-06-21", "2024-06-21"],
                "ask": [2.0, 2.5],
                "underlyingPrice": [100.0, 98.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path), "--strike", "90"]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "loss"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            rc = module.main()

        self.assertEqual(rc, 0)
        fake_ax.yaxis.set_inverted.assert_called_once_with(True)
        fake_ax.set_ylabel.assert_called_once_with("loss %")

    def test_main_with_custom_loss_field_uses_custom_label(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL", "AAPL"],
                "optionSymbol": ["AAPL240621P00090000", "AAPL240621P00090000"],
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "strikePrice": [90.0, 90.0],
                "maturityDate": ["2024-06-21", "2024-06-21"],
                "ask": [2.0, 2.5],
                "underlyingPrice": [100.0, 98.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path), "--strike", "90"]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "loss@30%"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            rc = module.main()

        self.assertEqual(rc, 0)
        fake_ax.set_title.assert_any_call("AAPL · Strike 90.0 · Expiry 2024-06-21 · loss@30%")

    def test_main_adds_underlying_overlay_when_available(self) -> None:
        quotes = pd.DataFrame(
            {
                "underlying": ["AAPL", "AAPL"],
                "updated": pd.to_datetime(["2024-03-01", "2024-03-02"]),
                "strikePrice": [100.0, 100.0],
                "maturityDate": ["2024-06-21", "2024-06-21"],
                "mid": [1.0, 1.1],
                "underlyingPrice": [100.0, 100.2],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "quotes.csv"
            quotes.to_csv(csv_path, index=False)

            fake_fig = MagicMock()
            fake_ax = MagicMock()
            overlay_sentinel = object()

            with patch("sys.argv", ["plot_option_prices.py", str(csv_path)]):
                with patch.dict(module.os.environ, {"PLOT_FIELD": "mid"}, clear=False):
                    with patch.object(module.plt, "subplots", return_value=(fake_fig, fake_ax)):
                        with patch.object(module.plt, "show"):
                            with patch.object(
                                module,
                                "add_underlying_overlay",
                                return_value=overlay_sentinel,
                            ) as overlay_mock:
                                rc = module.main()

        self.assertEqual(rc, 0)
        overlay_mock.assert_called_once()
        self.assertIs(fake_fig._underlying_overlay, overlay_sentinel)


if __name__ == "__main__":
    unittest.main()