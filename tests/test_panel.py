import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import fdi_qol as q


def _series(iso, years, value):
    return pd.DataFrame({"iso2": iso, "year": list(years), "value": value})


def _inputs(fdi_by_iso):
    isos = list(q.CT.iso2)
    mk = lambda v: pd.concat([_series(i, [2024], v(k)) for k, i in enumerate(isos)])
    income, sat, pop = mk(lambda k: 10000 + 500 * k), mk(lambda k: 6 + 0.05 * k), mk(lambda k: 1e6)
    fdi = pd.concat([_series(i, ys, 20 + k) for k, (i, ys) in enumerate(fdi_by_iso.items())])
    return income, sat, fdi, pop


def _fdi_full(**override):
    base = {i: range(2020, 2025) for i in q.CT.iso2}
    base.update(override)
    return base


def _panel(**override):
    return q.make_panel(*_inputs(_fdi_full(**override))).set_index("iso2")


def test_full_coverage_is_ok():
    p = _panel()
    assert p.fdi_ok.all()
    assert (p.fdi_last_year == 2024).all() and (p.fdi_years == 5).all()


def test_single_old_year_is_not_ok():
    p = _panel(GB=[2016])
    assert not p.loc["GB", "fdi_ok"]
    assert p.loc["GB", "fdi_last_year"] == 2016
    assert not np.isnan(p.loc["GB", "fdi"])  # value is kept


def test_stale_window_is_not_ok():
    assert not _panel(MT=range(2013, 2017)).loc["MT", "fdi_ok"]


def test_gappy_window_is_not_ok():
    assert not _panel(RS=[2013, 2014, 2022]).loc["RS", "fdi_ok"]


def test_not_ok_country_has_no_bivariate_class():
    p = _panel(GB=[2016])
    assert pd.isna(p.loc["GB", "cls_income"]) or p.loc["GB", "cls_income"] is None
    assert p.loc["DE", "cls_income"] is not None


def test_not_ok_country_excluded_from_tercile_cuts():
    # an extreme not-ok value must not move the cut points
    a = _panel()
    b = _panel(GB=[2016])
    b_fdi = _inputs(_fdi_full(GB=[2016]))[2]
    assert (b_fdi.loc[b_fdi.iso2 == "GB", "value"] < 1000).all()
    assert (a.fdi_t.drop("GB") == b.fdi_t.drop("GB")).all()


def test_gap_before_recent_run_is_ok():
    # Iceland-like: old points, then a 4-year recent run. Window stays recent enough.
    assert _panel(IS=[2016, 2021, 2022, 2023, 2024]).loc["IS", "fdi_ok"]


def test_mostly_old_points_with_one_recent_is_not_ok():
    # Austria-like: 2016-2019 plus 2024. Average mixes a decade apart.
    assert not _panel(AT=[2016, 2017, 2018, 2019, 2024]).loc["AT", "fdi_ok"]


def test_non_finite_values_do_not_count_as_points():
    ins = _inputs(_fdi_full())
    fdi = ins[2]
    bad = pd.DataFrame({"iso2": "FR", "year": [2022, 2023, 2024], "value": [np.nan, np.inf, np.nan]})
    fdi = pd.concat([fdi[~((fdi.iso2 == "FR") & (fdi.year >= 2022))], bad])
    p = q.make_panel(ins[0], ins[1], fdi, ins[3]).set_index("iso2")
    assert np.isfinite(p.loc["FR", "fdi"])
    assert p.loc["FR", "fdi_years"] == 2  # only 2020, 2021 are real
    assert not p.loc["FR", "fdi_ok"]


def test_min_points_boundary():
    assert _panel(FR=[2016, 2017, 2022, 2023, 2024]).loc["FR", "fdi_ok"]  # exactly 3 recent
    assert not _panel(FR=[2016, 2017, 2018, 2023, 2024]).loc["FR", "fdi_ok"]  # 2 recent
    # year == latest-5 (2019) is outside the window: only 2020 and 2021 are recent here
    assert not _panel(FR=[2015, 2016, 2019, 2020, 2021]).loc["FR", "fdi_ok"]


def test_country_missing_from_feed_is_not_ok():
    base = _fdi_full()
    del base["DE"]
    p = q.make_panel(*_inputs(base)).set_index("iso2")
    assert not p.loc["DE", "fdi_ok"]


def test_make_panel_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "DATA", tmp_path)
    _panel()
    assert list(tmp_path.iterdir()) == []
