import io
import os
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import download_option_data as module


class OptionalIntEnvTests(unittest.TestCase):
    def test_missing_optional_int_returns_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(module._optional_int_env("MIN_VOLUME"))

    def test_optional_int_accepts_zero(self) -> None:
        with patch.dict(os.environ, {"MIN_VOLUME": "0"}, clear=True):
            self.assertEqual(module._optional_int_env("MIN_VOLUME"), 0)


class QuoteColumnTests(unittest.TestCase):
    def test_missing_optional_columns_are_padded(self) -> None:
        columns = module._quote_columns(
            {"updated": [1, 2], "bid": [10.0, 11.0]},
            ["updated", "bid", "ask"],
            "OPT",
        )

        self.assertEqual(columns["ask"], [None, None])

    def test_mismatched_column_lengths_raise(self) -> None:
        with self.assertRaises(module.MarketDataError):
            module._quote_columns(
                {"updated": [1, 2], "bid": [10.0]},
                ["updated", "bid"],
                "OPT",
            )


class FetchQuoteHistoryTests(unittest.TestCase):
    def test_fetch_quote_history_adds_strike_and_maturity(self) -> None:
        response = {
            "s": "ok",
            "updated": [1710460800],
            "bid": [1.25],
            "ask": [1.5],
        }

        with patch.object(module, "_get", return_value=response):
            rows = module.fetch_quote_history(
                "token",
                "AAPL240315P00150000",
                "2024-03-01",
                "2024-03-16",
                underlying="AAPL",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["underlying"], "AAPL")
        self.assertEqual(rows[0]["strikePrice"], 150.0)
        self.assertEqual(rows[0]["maturityDate"], "2024-03-15")
        self.assertEqual(rows[0]["optionSymbol"], "AAPL240315P00150000")
        self.assertIsNone(rows[0]["bidSize"])


class GetTests(unittest.TestCase):
    def test_get_raises_latest_available_session_error_for_closed_session_cap(self) -> None:
        response = MagicMock()
        response.status_code = 402
        response.text = (
            '{"s":"error","errmsg":"Your plan can only access fully-closed '
            'sessions; the latest available is 2026-06-12."}'
        )
        response.raise_for_status.side_effect = module.requests.HTTPError(response=response)

        session = MagicMock()
        session.get.return_value = response

        with self.assertRaises(module.LatestAvailableSessionError) as exc:
            module._get("https://example.test/options/chain/QQQ/", "token", session=session)

        self.assertEqual(exc.exception.latest_available_date, "2026-06-12")


class FetchChainSymbolsTests(unittest.TestCase):
    def test_fetch_chain_symbols_rejects_non_string_symbols(self) -> None:
        with patch.object(module, "_get", return_value={"s": "ok", "optionSymbol": ["A", 42]}):
            with self.assertRaises(module.MarketDataError):
                module.fetch_chain_symbols("token", "AAPL", "put", "2024-01-01", "2024-01-02")


class CollectChainSymbolsTests(unittest.TestCase):
    def test_collect_chain_symbols_walks_backward_and_unions_symbols(self) -> None:
        snapshots = {
            "2024-03-20": ["X", "Y"],
            "2024-03-13": ["Y", "Z"],
            "2024-03-06": [],
        }

        def fake_fetch_chain_symbols(
            token: str,
            symbol: str,
            side: str,
            exp_from: str,
            exp_to: str,
            snapshot_date: str | None = None,
            extra: dict[str, object] | None = None,
            session: object | None = None,
        ) -> list[str]:
            self.assertEqual(token, "token")
            self.assertEqual(symbol, "AAPL")
            self.assertEqual(side, "put")
            self.assertEqual(exp_from, "2024-03-01")
            self.assertEqual(exp_to, "2024-03-15")
            self.assertEqual(extra, {"delta": "0.25"})
            self.assertIsNone(session)
            return snapshots[snapshot_date or ""]

        with patch.object(module, "fetch_chain_symbols", side_effect=fake_fetch_chain_symbols) as fetch_chain_mock:
            with patch("builtins.print"):
                symbols = module.collect_chain_symbols(
                    "token",
                    "AAPL",
                    "put",
                    "2024-03-01",
                    "2024-03-01",
                    "2024-03-15",
                    "2024-03-20",
                    7,
                    extra={"delta": "0.25"},
                    today=date(2024, 3, 20),
                )

        self.assertEqual(symbols, ["X", "Y", "Z"])
        self.assertEqual(
            [call.args[5] for call in fetch_chain_mock.call_args_list],
            ["2024-03-20", "2024-03-13", "2024-03-06"],
        )

    def test_collect_chain_symbols_uses_quote_latest_date_when_expiry_is_later(self) -> None:
        with patch.object(module, "fetch_chain_symbols", return_value=["X", "Y"]) as fetch_chain_mock:
            with patch("builtins.print"):
                symbols = module.collect_chain_symbols(
                    "token",
                    "AAPL",
                    "put",
                    "2024-01-02",
                    "2024-03-01",
                    "2024-03-31",
                    "2024-01-04",
                    7,
                    today=date(2026, 6, 14),
                )

        self.assertEqual(symbols, ["X", "Y"])
        self.assertEqual(fetch_chain_mock.call_count, 1)
        self.assertEqual(fetch_chain_mock.call_args.args[5], "2024-01-04")

    def test_collect_chain_symbols_rejects_future_quote_window(self) -> None:
        with patch.object(module, "fetch_chain_symbols") as fetch_chain_mock:
            with self.assertRaises(module.MarketDataError) as exc:
                module.collect_chain_symbols(
                    "token",
                    "QQQ",
                    "put",
                    "2026-10-01",
                    "2026-10-01",
                    "2026-12-31",
                    "2026-10-31",
                    7,
                    today=date(2026, 6, 14),
                )

        self.assertIn("QUOTE_LATEST_DATE is the last quote day to include", str(exc.exception))
        fetch_chain_mock.assert_not_called()

class ConfigTests(unittest.TestCase):
    def test_quote_latest_date_defaults_to_today(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_DAYS_BACK": "14",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-15",
        }

        class FakeDate(date):
            @classmethod
            def today(cls) -> "FakeDate":
                return cls(2024, 3, 15)

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.object(module, "date", FakeDate):
                with patch.dict(os.environ, env, clear=True):
                    config = module.Config.from_env()

        self.assertEqual(config.quote_latest_date, "2024-03-15")
        self.assertEqual(config.quote_start_date, "2024-03-01")
        self.assertEqual(config.quote_end_date_exclusive, "2024-03-16")

    def test_quote_window_is_derived_from_latest_date_and_days_back(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_LATEST_DATE": "2024-03-15",
            "QUOTE_DAYS_BACK": "14",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-15",
        }

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.dict(os.environ, env, clear=True):
                config = module.Config.from_env()

        self.assertEqual(config.quote_latest_date, "2024-03-15")
        self.assertEqual(config.quote_days_back, 14)
        self.assertEqual(config.quote_start_date, "2024-03-01")
        self.assertEqual(config.quote_end_date_exclusive, "2024-03-16")

    def test_write_chunk_files_defaults_to_false(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_LATEST_DATE": "2024-03-15",
            "QUOTE_DAYS_BACK": "14",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-15",
        }

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.dict(os.environ, env, clear=True):
                config = module.Config.from_env()

        self.assertFalse(config.write_chunk_files)

    def test_lookback_step_days_must_be_positive(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_LATEST_DATE": "2024-03-15",
            "QUOTE_DAYS_BACK": "14",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-15",
            "LOOKBACK_STEP_DAYS": "0",
        }

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as exc:
                    module.Config.from_env()

        self.assertEqual(str(exc.exception), "error: LOOKBACK_STEP_DAYS must be a positive integer")

    def test_quote_days_back_must_be_non_negative(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_LATEST_DATE": "2024-03-15",
            "QUOTE_DAYS_BACK": "-1",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-15",
        }

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as exc:
                    module.Config.from_env()

        self.assertEqual(
            str(exc.exception),
            "error: QUOTE_DAYS_BACK must be zero or a positive integer",
        )

    def test_exp_to_must_be_iso_date(self) -> None:
        env = {
            "MARKETDATA_TOKEN": "token",
            "SYMBOL": "AAPL",
            "SIDE": "put",
            "QUOTE_LATEST_DATE": "2024-03-15",
            "QUOTE_DAYS_BACK": "14",
            "EXP_FROM": "2024-03-15",
            "EXP_TO": "2024-03-33",
        }

        with patch.object(module, "load_dotenv", return_value=None):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(SystemExit) as exc:
                    module.Config.from_env()

        self.assertEqual(str(exc.exception), "error: EXP_TO must be YYYY-MM-DD")


class MainTests(unittest.TestCase):
    def test_main_retries_with_latest_closed_session_date(self) -> None:
        config = module.Config(
            token="token",
            symbol="QQQ",
            side="put",
            quote_latest_date="2026-06-15",
            quote_days_back=30,
            exp_from="2026-10-01",
            exp_to="2026-12-31",
            lookback_step_days=7,
            out_dir_root=Path("data"),
            write_chunk_files=False,
            extra={},
        )
        row = {
            "optionSymbol": "QQQ261016P00300000",
            "strikePrice": 300.0,
            "maturityDate": "2026-10-16",
            "updated": 1781568000,
        }
        session = MagicMock()
        session_factory = MagicMock()
        session_factory.__enter__.return_value = session
        session_factory.__exit__.return_value = False

        with patch.object(module.Config, "from_env", return_value=config):
            with patch.object(module.requests, "Session", return_value=session_factory):
                with patch.object(
                    module,
                    "collect_chain_symbols",
                    side_effect=[
                        module.LatestAvailableSessionError(
                            f"{module.API_ROOT}/options/chain/QQQ/",
                            "2026-06-12",
                            (
                                '{"s":"error","errmsg":"Your plan can only access '
                                'fully-closed sessions; the latest available is '
                                '2026-06-12."}'
                            ),
                        ),
                        [row["optionSymbol"]],
                    ],
                ) as collect_chain_mock:
                    with patch.object(module, "fetch_quote_history", return_value=[row]) as fetch_quote_history_mock:
                        with patch.object(module, "write_csv") as write_csv_mock:
                            with patch("builtins.print") as print_mock:
                                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                                    exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(collect_chain_mock.call_count, 2)
        self.assertEqual(collect_chain_mock.call_args_list[0].args[6], "2026-06-15")
        self.assertEqual(collect_chain_mock.call_args_list[1].args[6], "2026-06-12")
        self.assertEqual(fetch_quote_history_mock.call_args.args[2], "2026-05-13")
        self.assertEqual(fetch_quote_history_mock.call_args.args[3], "2026-06-13")
        print_mock.assert_any_call(
            "warning: marketdata.app plan only allows fully-closed sessions; "
            "retrying with QUOTE_LATEST_DATE=2026-06-12",
            file=stderr,
        )
        self.assertEqual(write_csv_mock.call_count, 1)

    def test_main_skips_per_contract_files_when_chunk_writes_disabled(self) -> None:
        config = module.Config(
            token="token",
            symbol="AAPL",
            side="put",
            quote_latest_date="2024-03-15",
            quote_days_back=14,
            exp_from="2024-03-15",
            exp_to="2024-03-15",
            lookback_step_days=7,
            out_dir_root=Path("data"),
            write_chunk_files=False,
            extra={},
        )
        row = {
            "optionSymbol": "AAPL240315P00150000",
            "strikePrice": 150.0,
            "maturityDate": "2024-03-15",
            "updated": 1710460800,
        }
        session = MagicMock()
        session_factory = MagicMock()
        session_factory.__enter__.return_value = session
        session_factory.__exit__.return_value = False

        with patch.object(module.Config, "from_env", return_value=config):
            with patch.object(module.requests, "Session", return_value=session_factory):
                with patch.object(module, "collect_chain_symbols", return_value=[row["optionSymbol"]]):
                    with patch.object(module, "fetch_quote_history", return_value=[row]):
                        with patch.object(module, "write_csv") as write_csv_mock:
                            with patch("builtins.print"):
                                exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(write_csv_mock.call_count, 1)
        self.assertEqual(
            write_csv_mock.call_args.args[0],
            Path("data") / "AAPL" / "put" / "all_puts.csv",
        )


if __name__ == "__main__":
    unittest.main()