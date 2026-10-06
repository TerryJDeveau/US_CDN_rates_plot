#!/usr/bin/env python3
"""Check the web page (``ratesplot/web.py``): its charts against the command line's, and how it behaves.

Two parts, both run by default:

* ``charts``: for a few regression cases, the page's own drawing is hashed
  against the command line's. The page reads the case as it reads its
  address (``cli.sort_tokens``, then ``frontend.starting_choices``) and turns
  its controls' choices into a config (``web._config_for``), which must draw
  what the command line's does (``options.drawn_config`` for each nation); then it draws with ``web._draw``. The command line
  renders the same case through ``tools/verify_charts.run`` into
  ``out/verify_web/NAME/``. Each chart is IDENTICAL or DIFFER: on the same
  machine the page must draw, byte for byte, what the command line and the
  window draw.
* ``page``: Streamlit's ``AppTest`` runs ``streamlit_app.py`` headless in this
  process and drives it as a visitor would: a chart in the address, the rules
  between controls, a bad field, Reset, a bad address, the date buttons with
  Back and Unzoom, the calendars, the size boxes and their aspect lock. Each
  check prints ``ok`` or ``FAIL``.

A drag on a chart is not tested here: AppTest runs no JavaScript, so the chart
component never sends a box. The rule a box goes through,
``frontend.zoom_updates``, is the window's own; test a drag in a browser.

Downloads come from the regression tool's disk cache (``--cache``, default
``out/download_cache``, shared with ``tools/regress_charts.py``), and the
charts part pins ``--e:2026-09-01`` as that tool does. Several page checks
draw the default charts, which end today: their intraday quotes are fetched
live, so the page part needs the network.

Usage (from the project root)::

    python tools/verify_web.py                 # both parts (a few minutes once the cache is warm)
    python tools/verify_web.py --only charts   # or --only page
    python tools/verify_web.py --list          # the chart cases
    python tools/verify_web.py --synthetic     # made-up data, no network (tools/synthetic_sources.py)

Exit code 1 on any DIFFER or FAIL.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import logging
import sys
from pathlib import Path
from typing import Callable

# The page's buttons are labelled "◀ Back" and "Later ▶"; a console or pipe
# in the Windows code page cannot encode them. Escape what it cannot show
# rather than fail. (Before ratesplot is imported: its output router writes
# through the stream that is sys.stdout now.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import verify_charts  # noqa: E402  (first: it sets the Agg backend before pyplot is imported)
import pandas as pd  # noqa: E402
import synthetic_sources  # noqa: E402
from regress_charts import CASES, PINNED_END, install_disk_cache  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from ratesplot import http, web  # noqa: E402
from ratesplot.cli import sort_tokens  # noqa: E402
from ratesplot.frontend import move_dates_updates, starting_choices  # noqa: E402
from ratesplot.options import command_line_tokens, config_from_choices, drawn_config  # noqa: E402

# AppTest touches Streamlit's state before each page's first run, outside any
# script run, and Streamlit logs "missing ScriptRunContext! This warning can be
# ignored when running in bare mode" each time, which it is here. A filter,
# not a level: Streamlit resets its loggers' levels when a page runs.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").addFilter(
    lambda record: "missing ScriptRunContext" not in record.getMessage()
)

APP = str(ROOT / "streamlit_app.py")
TIMEOUT_SECONDS = 180  # per AppTest run: a drawing whose downloads are not cached yet takes about 15 s

# Regression cases drawn both ways: the defaults, each measure, levels, -l,
# --reg with and without a tolerance, the history to 1867, a small canvas.
CHART_CASES = (
    "default", "c2015", "u_reg_r", "c_lv_r", "c_l_lv", "u_reg_tol3", "c_reg1867", "c_p", "c1100",
    # Batch 1 (2026-10-06): --no-yields read from the address, and the yield axis's other curves.
    "u_noy", "all_rates",
    # Batch 2: values for one nation's chart.
    "nat_limits", "nat_terms",
)

VIEW_BUTTONS = ("◀ Back", "Unzoom", "◀ Earlier", "Later ▶", "Zoom out")

Check = Callable[..., None]


# ---------------------------------------------------------------------------
# Part 1: the page's charts against the command line's
# ---------------------------------------------------------------------------


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def check_charts(cases: tuple[str, ...], outdir: Path, verbose: bool = False) -> bool:
    """Draw each case as the page does and as the command line does; True if every chart is identical."""
    all_same = True
    for name in cases:
        tokens = [PINNED_END, *CASES[name]]
        payloads, flags, unmatched, misspelled = sort_tokens(tokens)
        if unmatched or misspelled:
            print(f"{name:12} FAIL: the case does not parse ({unmatched + misspelled})")
            all_same = False
            continue
        # As the page does: the address laid over the defaults, then the controls' choices to a config.
        config = config_from_choices(payloads, flags)
        choices, _notes = starting_choices(config, payloads, flags, {})
        page_config, problems = web._config_for(choices)
        if page_config is None or any(drawn_config(page_config, k) != drawn_config(config, k) for k in ("cdn", "us")):
            print(f"{name:12} FAIL: the page's config differs from the command line's ({problems})")
            all_same = False
            continue
        drawing = web._draw(page_config, tuple(command_line_tokens(*choices)))
        if drawing.error is not None:
            print(f"{name:12} FAIL: the page's drawing failed: {drawing.error}")
            all_same = False
            continue
        output = io.StringIO()
        with contextlib.nullcontext() if verbose else contextlib.redirect_stdout(output):
            written = verify_charts.run(outdir / name, tokens)
        keys = [key for key in ("cdn", "us") if key in drawing.pngs]
        verdicts = []
        if len(keys) != len(written):
            verdicts.append(f"FAIL: the page drew {len(keys)} chart(s), the command line {len(written)}")
            all_same = False
        for key, path in zip(keys, written):
            same = sha256(drawing.pngs[key]) == sha256(path.read_bytes())
            verdicts.append(f"{key}:{'IDENTICAL' if same else 'DIFFER'}")
            all_same &= same
        print(f"{name:12} {' '.join(CASES[name]):28} {' '.join(verdicts)}  (page {drawing.seconds:.1f} s)")
        # The command line's warnings (data coverage, fallbacks), as tools/regress_charts.py echoes them.
        for line in output.getvalue().splitlines():
            if line.strip().startswith("Warning:") and "requested start" not in line:
                print(f"    {line.strip()}")
    return all_same


# ---------------------------------------------------------------------------
# Part 2: the page's behaviour, through Streamlit's AppTest
# ---------------------------------------------------------------------------


def new_page(chart: str | None = None) -> AppTest:
    """A fresh visitor's page, with ``chart`` in its address (None: no address), run once."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT_SECONDS)
    if chart is not None:
        at.query_params["chart"] = chart
    at.run()
    return at


def flags(at: AppTest) -> dict[str, bool]:
    return {box.key.split(":", 1)[1]: box.value for box in at.checkbox if box.key and box.key.startswith("flag:")}


def charts_shown(at: AppTest) -> int:
    """The charts shown: each is the page's chart component (a ``bidi_component`` to AppTest)."""
    return len(at.get("bidi_component"))


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def enabled(at: AppTest) -> dict[str, bool]:
    return {b.label: not b.disabled for b in at.button if b.label in VIEW_BUTTONS}


def address(at: AppTest) -> str:
    """The chart's command line in the page's address (AppTest may hold a query value as a list)."""
    value = at.query_params.get("chart", "")
    return value[0] if isinstance(value, list) else value


def png_size_bytes(width: int, height: int) -> bytes:
    """A PNG header's width and height (bytes 16-24), for comparing with a drawn chart."""
    return width.to_bytes(4, "big") + height.to_bytes(4, "big")


def _is_automatic_date(placeholder: str) -> bool:
    """``automatic: YYYY-MM-DD``, any date (the year follows config.DEFAULT_START_FLOOR)."""
    prefix = "automatic: "
    if not placeholder.startswith(prefix):
        return False
    try:
        datetime.date.fromisoformat(placeholder.removeprefix(prefix))
    except ValueError:
        return False
    return True


def check_address_and_rules(check: Check) -> None:
    """A chart in the address; the rules between controls; a bad field; levels and tolerance; Reset."""
    at = new_page("-c -r --reg --e:2026-09-01")
    check("no exception", not at.exception, at.exception)
    f = flags(at)
    check("address: Canada only, -r, --reg", f["cdn"] and not f["us"] and f["relative"] and f["regression"], f)
    check("GDP box disabled under -r", at.checkbox(key="flag:gdp").disabled)
    check("tolerance field enabled with --reg", not at.text_input(key="value:regression-tolerance").disabled)
    check("one chart drawn", charts_shown(at) == 1, charts_shown(at))
    check("address kept", address(at) == "--C --relative --regression --end:2026-09-01", address(at))
    code = [c.value for c in at.code]
    check("command line shown", any(v.startswith("python US_CDN_rates_plot.py --C --relative") for v in code), code[:1])
    placeholder = at.text_input(key="value:start").placeholder
    check("start placeholder shows the date found", _is_automatic_date(placeholder), placeholder)

    # One measure at a time; changing it clears Top and Bottom.
    at.text_input(key="value:cdn:top").input("200%")
    at.run()
    at.checkbox(key="flag:per-capita").check()
    at.run()
    f = flags(at)
    check("ticking -p unticks -r", f["per-capita"] and not f["relative"], f)
    top = at.text_input(key="value:cdn:top").value
    check("Top cleared by the measure change", top == "", top)
    check("address follows", address(at) == "--C --per-capita --regression --end:2026-09-01", address(at))

    # The last country cannot be unticked.
    at.checkbox(key="flag:cdn").uncheck()
    at.run()
    check("Canada ticked again", flags(at)["cdn"])
    check("notice shown", any("At least one country" in i.value for i in at.info), [i.value for i in at.info])

    # A bad field: not drawn, the previous chart stays, the address is not changed.
    at.text_input(key="value:start").input("20x5")
    at.run()
    check("error for a bad start", any("Not drawn" in e.value for e in at.error), [e.value for e in at.error])
    check("previous chart still shown", charts_shown(at) == 1)
    check("address unchanged", address(at) == "--C --per-capita --regression --end:2026-09-01", address(at))
    at.text_input(key="value:start").input("2001")
    at.run()
    check(
        "drawn again from 2001",
        not at.error and address(at) == "--C --per-capita --regression --start:2001 --end:2026-09-01",
        (address(at), [e.value for e in at.error]),
    )

    # The level boxes and the tolerance.
    at.checkbox(key="level:f").check()
    at.checkbox(key="level:p").check()
    at.text_input(key="value:regression-tolerance").input("3")
    at.run()
    check("levels and tolerance in the address", "--debt:fp" in address(at) and "--reg:3" in address(at), address(at))
    at.checkbox(key="flag:regression").uncheck()
    at.run()
    check(
        "tolerance greyed and left out when --reg is off",
        at.text_input(key="value:regression-tolerance").disabled and "reg" not in address(at),
        address(at),
    )

    # Reset.
    button(at, "Reset").click()
    at.run()
    f = flags(at)
    check("reset: both countries, no measure", f["cdn"] and f["us"] and not f["relative"] and not f["per-capita"], f)
    check("reset: two charts", charts_shown(at) == 2, charts_shown(at))


def check_bad_addresses(check: Check) -> None:
    """An address the program cannot read: reported, and the defaults drawn."""
    bad = new_page("pizza -r:1 --e:2026-09-01")
    warnings = [w.value for w in bad.warning]
    check("bad address reported", any("pizza: not an option" in w and "takes no value" in w for w in warnings), warnings)
    check("defaults drawn", charts_shown(bad) == 2, charts_shown(bad))
    # Contradictory options (-r with -p): reported by config_from_choices.
    both = new_page("-r -p --e:2026-09-01")
    check("-r -p reported", bool(both.warning), [w.value for w in both.warning])


def check_view_buttons(check: Check) -> None:
    """Back, Unzoom and the date buttons, on a pinned window; Reset undone by Back; Later off at today."""
    view = new_page("-c -s:2001 --e:2019")
    expected = {"◀ Back": False, "Unzoom": False, "◀ Earlier": True, "Later ▶": True, "Zoom out": True}
    check("view: nothing to undo; the dates can move", enabled(view) == expected, enabled(view))
    png = view.get("bidi_component")[0].proto.bytes
    check(
        "the chart component gets the PNG, 2048x1536",
        png[:8] == b"\x89PNG\r\n\x1a\n" and png[16:24] == png_size_bytes(2048, 1536),
        len(png),
    )

    def expect(start: str, end: str, **move: float) -> tuple[str, str, str]:
        """The address a date move should give (by ``frontend.move_dates_updates``), and the new start and end."""
        u = move_dates_updates(pd.Timestamp(start), pd.Timestamp(end), **move)
        return f"--C --start:{u['start']}" + (f" --end:{u['end']}" if u["end"] else ""), u["start"], u["end"]

    e1, s1, n1 = expect("2001-01-01", "2019-01-01", shift=-0.5)
    button(view, "◀ Earlier").click()
    view.run()
    check("Earlier: half a window back", address(view) == e1, (address(view), e1))
    check("Back and Unzoom now possible", enabled(view)["◀ Back"] and enabled(view)["Unzoom"], enabled(view))
    e2, _s2, _n2 = expect(s1, n1, scale=2.0)
    button(view, "Zoom out").click()
    view.run()
    check("Zoom out: twice the span", address(view) == e2, (address(view), e2))
    button(view, "◀ Back").click()
    view.run()
    check("Back: one step", address(view) == e1, address(view))
    e3, _s3, _n3 = expect(s1, n1, shift=0.5)
    button(view, "Later ▶").click()
    view.run()
    check("Later: half a window on", address(view) == e3, (address(view), e3))
    button(view, "Unzoom").click()
    view.run()
    # Earlier then Later came back to the same dates, so the fields as first
    # typed are the same chart: nothing is redrawn and the address keeps "2001".
    fields = (view.text_input(key="value:start").value, view.text_input(key="value:end").value)
    check(
        "Unzoom: the fields as first typed, nothing left to undo",
        fields == ("2001", "2019") and not enabled(view)["◀ Back"],
        (fields, enabled(view)),
    )
    button(view, "Reset").click()
    view.run()
    check("Reset draws the defaults", address(view) == "" and charts_shown(view) == 2, address(view))
    button(view, "◀ Back").click()
    view.run()
    check("Back undoes Reset", address(view) == "--C --start:2001 --end:2019", address(view))
    today = new_page()
    check("at today, Later is off", not enabled(today)["Later ▶"] and enabled(today)["◀ Earlier"], enabled(today))


def check_calendars(check: Check) -> None:
    """The calendar beside a date field: it opens at the field's date and writes a picked day into it."""
    cal = new_page("-c -s:2001-03 --e:2026-09-01")
    opened = cal.date_input(key="calendar:start").value
    check("calendar opens at the field's date", opened == datetime.date(2001, 3, 1), opened)
    cal.date_input(key="calendar:start").set_value(datetime.date(2005, 6, 15))
    cal.run()
    check(
        "a picked day is written in the field",
        cal.text_input(key="value:start").value == "2005-06-15" and "--start:2005-06-15" in address(cal),
        address(cal),
    )
    cal.button(key="blank:start").click()
    cal.run()
    check("Blank empties the field", cal.text_input(key="value:start").value == "" and "--start" not in address(cal), address(cal))
    cal.text_input(key="value:start").input("1700")
    cal.run()
    check(
        "a date outside the calendar leaves it empty, no error",
        cal.date_input(key="calendar:start").value is None and not cal.exception,
        cal.date_input(key="calendar:start").value,
    )


def check_size_boxes(check: Check) -> None:
    """Width and Height with the aspect-ratio lock: the WxH text stays the source of truth."""

    def boxes(at: AppTest) -> tuple[str, str]:
        return at.text_input(key="size:width").value, at.text_input(key="size:height").value

    size = new_page("-c -s:2015 --e:2026-09-01 -dim:1600x900")
    check(
        "boxes show the address's size; the lock is on",
        boxes(size) == ("1600", "900") and size.checkbox(key="size:lock").value,
        boxes(size),
    )
    size.text_input(key="size:width").input("2400")
    size.run()
    check(
        "locked: width 2400 gives height 1350, drawn",
        boxes(size) == ("2400", "1350")
        and "--dimensions:2400x1350" in address(size)
        and size.get("bidi_component")[0].proto.bytes[16:24] == png_size_bytes(2400, 1350),
        (boxes(size), address(size)),
    )
    size.text_input(key="size:height").input("1000")
    size.run()
    check("locked: height 1000 gives width 1778 (the loaded 16:9, no drift)", boxes(size) == ("1778", "1000"), boxes(size))
    size.checkbox(key="size:lock").uncheck()
    size.run()
    size.text_input(key="size:width").input("2000")
    size.run()
    check(
        "unlocked: the height stays",
        boxes(size) == ("2000", "1000") and "--dimensions:2000x1000" in address(size),
        (boxes(size), address(size)),
    )
    size.checkbox(key="size:lock").check()
    size.run()
    size.text_input(key="size:height").input("900")
    size.run()
    check("locked again: the shape of when it was ticked (2:1)", boxes(size) == ("1800", "900"), boxes(size))
    button(size, "Reset").click()
    size.run()
    check(
        "Reset: the default size in the boxes",
        boxes(size) == ("2048", "1536") and "dimensions" not in address(size),
        (boxes(size), address(size)),
    )
    button(size, "◀ Back").click()
    size.run()
    check(
        "Back: the boxes follow",
        boxes(size) == ("1800", "900") and "--dimensions:1800x900" in address(size),
        (boxes(size), address(size)),
    )
    one = new_page("-c -s:2015 --e:2026-09-01 -dim:1100")
    check(
        "one dimension: the implied 4:3 shown, the address as given",
        boxes(one) == ("1100", "825") and "--dimensions:1100" in address(one) and "1100x" not in address(one),
        (boxes(one), address(one)),
    )
    one.text_input(key="size:width").input("12a")
    one.run()
    check("a bad box: reported, not drawn", any("Size" in e.value for e in one.error), [e.value for e in one.error])


def check_choice_boxes(check: Check) -> None:
    """The tick boxes of the list options (yield terms, mortgage terms, spreads): the text stays the source of truth."""

    def ticked(at: AppTest, name: str) -> list[str]:
        prefix = f"choice:{name}:"
        return [box.key[len(prefix):] for box in at.checkbox if box.key and box.key.startswith(prefix) and box.value]

    page = new_page("-u --e:2026-09-01 --no-y:30 --mo:15 --sp:10-2,7-1m")
    check(
        "the address's lists ticked; a pair not offered gets its own box",
        ticked(page, "us:yields") == ["3m", "2y", "5y", "10y"]
        and ticked(page, "us:mortgage-terms") == ["15"]
        and ticked(page, "us:spread-pairs") == ["10y-2y", "7y-1m"],
        (ticked(page, "us:yields"), ticked(page, "us:mortgage-terms"), ticked(page, "us:spread-pairs")),
    )
    page.checkbox(key="choice:us:yields:7y").check()
    page.run()
    check("ticking 7y adds it to the address", "--yields:3m,2y,5y,7y,10y" in address(page), address(page))
    page.checkbox(key="choice:us:mortgage-terms:15").uncheck()
    page.run()
    check(
        "the last mortgage box cannot be unticked",
        ticked(page, "us:mortgage-terms") == ["15"] and any("untick “Mortgage rates”" in str(item.value) for item in list(page.info) + list(page.warning)),
        ticked(page, "us:mortgage-terms"),
    )
    page.checkbox(key="flag:spreads").uncheck()
    page.run()
    check(
        "spread boxes greyed while --spreads is off, and left out of the address",
        page.checkbox(key="choice:us:spread-pairs:10y-2y").disabled and "--spreads" not in address(page),
        address(page),
    )


def check_nation_sections(check: Check) -> None:
    """Batch 2: each nation's own section: only its terms, its own limits, and the address they write."""
    page = new_page("--e:2026-09-01 --us:max:9 --cdn:y:2,10 --mo")
    check("no exception", not page.exception, page.exception)
    keys = {box.key for box in page.checkbox if box.key}
    check(
        "Canada offers its five terms and its mortgage terms only",
        "choice:cdn:yields:30y" in keys and "choice:cdn:yields:7y" not in keys
        and "choice:cdn:mortgage-terms:5" in keys and "choice:cdn:mortgage-terms:30" not in keys,
        sorted(key for key in keys if key.startswith("choice:cdn:")),
    )
    check(
        "the address's values in each nation's fields",
        page.text_input(key="value:us:max").value == "9" and page.text_input(key="value:cdn:max").value == ""
        and page.checkbox(key="choice:cdn:yields:2y").value and not page.checkbox(key="choice:cdn:yields:30y").value
        and page.checkbox(key="choice:us:yields:30y").value,
        (page.text_input(key="value:us:max").value, page.text_input(key="value:cdn:max").value),
    )
    check("address kept", address(page) == "--mortgages --cdn:yields:2y,10y --end:2026-09-01 --us:max:9", address(page))
    page.text_input(key="value:cdn:max").input("9")
    page.run()
    check("the same value for both is written once", "--max:9" in address(page) and "--us:max" not in address(page), address(page))
    page.text_input(key="value:cdn:min").input("x")
    page.run()
    check("a bad nation field: reported with its nation", any("Canada, Min %" in e.value for e in page.error), [e.value for e in page.error])


PAGE_CHECKS = (
    check_address_and_rules, check_bad_addresses, check_view_buttons, check_calendars, check_size_boxes, check_choice_boxes,
    check_nation_sections,
)


def check_page() -> bool:
    """Run every page check; True if all passed."""
    results: list[bool] = []

    def check(label: str, condition: bool, detail: object = "") -> None:
        results.append(bool(condition))
        shown = f"  [{detail}]" if not condition or detail else ""
        print(f"{'ok  ' if condition else 'FAIL'} {label}{shown}", flush=True)

    for group in PAGE_CHECKS:
        print(f"-- {group.__doc__.splitlines()[0]}")
        group(check)
    print(f"[web] page: {sum(results)} of {len(results)} checks passed")
    return all(results)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("charts", "page"), help="run one part only")
    parser.add_argument("--cases", help=f"comma-separated regression cases for the charts part (default: {','.join(CHART_CASES)})")
    parser.add_argument("--cache", type=Path, help="download cache folder (default out/download_cache, or out/synthetic_cache)")
    parser.add_argument(
        "--synthetic", action="store_true", help="made-up data in each source's format, no network (tools/synthetic_sources.py)"
    )
    parser.add_argument("--verbose", action="store_true", help="show the command line's own output")
    parser.add_argument("--list", action="store_true", help="list the chart cases and exit")
    args = parser.parse_args(argv)
    cases = tuple(args.cases.split(",")) if args.cases else CHART_CASES
    if args.list:
        for name in cases:
            print(f"{name:12} {' '.join(CASES[name])}")
        return 0

    if args.synthetic:
        print("[web] SYNTHETIC data (tools/synthetic_sources.py): proves code paths, not the data")
        synthetic_sources.install_synthetic_cache(http, (args.cache or ROOT / "out" / "synthetic_cache").resolve())
    else:
        install_disk_cache(http, (args.cache or ROOT / "out" / "download_cache").resolve())
    passed = True
    if args.only in (None, "charts"):
        print(f"[web] charts: the page's drawing against the command line's, {len(cases)} case(s)")
        same = check_charts(cases, ROOT / "out" / "verify_web", verbose=args.verbose)
        print("[web] charts: ALL IDENTICAL" if same else "[web] charts: SOME DIFFER")
        passed &= same
    if args.only in (None, "page"):
        print("[web] page: Streamlit's AppTest")
        passed &= check_page()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
