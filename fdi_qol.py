#!/usr/bin/env python3
"""
FDI vs quality of life, Europe. Builds one interactive HTML file.
Usage, data sources and column meanings: see README.md.
"""
import argparse
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).parent
DATA = OUT / "data"
DATA.mkdir(exist_ok=True)

ES = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
WB = "https://api.worldbank.org/v2"
TOPO = "https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json"
SINCE = 2012
H = 420  # plot height, shared by scatter and map so each row aligns
SPARSE = "Sparse FDI data"
FDI_YEARS = 5  # average FDI over last N years with data
# FDI is "ok" (used in trend lines and tercile cuts) only if at least FDI_MIN_POINTS of the averaged
# points lie in the newest FDI_YEARS calendar years of the feed. Otherwise the average is stale or
# mixes years far apart (grey in the charts).
FDI_MIN_POINTS = 3

# name, iso2, iso numeric (map), eurostat code, group, SPE hub flag
C = [
    ("Austria", "AT", 40, "AT", "West", 0), ("Belgium", "BE", 56, "BE", "West", 0),
    ("Denmark", "DK", 208, "DK", "West", 0), ("Finland", "FI", 246, "FI", "West", 0),
    ("France", "FR", 250, "FR", "West", 0), ("Germany", "DE", 276, "DE", "West", 0),
    ("Greece", "GR", 300, "EL", "West", 0), ("Ireland", "IE", 372, "IE", "West", 1),
    ("Italy", "IT", 380, "IT", "West", 0), ("Luxembourg", "LU", 442, "LU", "West", 1),
    ("Netherlands", "NL", 528, "NL", "West", 1), ("Portugal", "PT", 620, "PT", "West", 0),
    ("Spain", "ES", 724, "ES", "West", 0), ("Sweden", "SE", 752, "SE", "West", 0),
    ("Norway", "NO", 578, "NO", "West", 0), ("Switzerland", "CH", 756, "CH", "West", 0),
    ("Iceland", "IS", 352, "IS", "West", 0), ("United Kingdom", "GB", 826, "UK", "West", 0),
    ("Cyprus", "CY", 196, "CY", "West", 1), ("Malta", "MT", 470, "MT", "West", 1),
    ("Bulgaria", "BG", 100, "BG", "East", 0), ("Croatia", "HR", 191, "HR", "East", 0),
    ("Czechia", "CZ", 203, "CZ", "East", 0), ("Estonia", "EE", 233, "EE", "East", 0),
    ("Hungary", "HU", 348, "HU", "East", 0), ("Latvia", "LV", 428, "LV", "East", 0),
    ("Lithuania", "LT", 440, "LT", "East", 0), ("Poland", "PL", 616, "PL", "East", 0),
    ("Romania", "RO", 642, "RO", "East", 0), ("Slovakia", "SK", 703, "SK", "East", 0),
    ("Slovenia", "SI", 705, "SI", "East", 0), ("Serbia", "RS", 688, "RS", "East", 0),
    ("Albania", "AL", 8, "AL", "East", 0), ("North Macedonia", "MK", 807, "MK", "East", 0),
    ("Montenegro", "ME", 499, "ME", "East", 0),
]
CT = pd.DataFrame(C, columns=["country", "iso2", "iso_num", "es", "group", "spe"])
CT["spe"] = CT["spe"].map({1: "SPE hub", 0: "other"})
ES2ISO = dict(zip(CT.es, CT.iso2))


# ---------- Eurostat JSON-stat ----------
def _get(url, params=None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _parse(js):
    ids, sizes = js["id"], js["size"]
    cats = {d: sorted(js["dimension"][d]["category"]["index"].items(), key=lambda kv: kv[1])
            for d in ids}
    cats = {d: [k for k, _ in v] for d, v in cats.items()}
    val = js["value"]
    items = val.items() if isinstance(val, dict) else enumerate(val)
    rows = []
    for k, v in items:
        if v is None:
            continue
        idx = np.unravel_index(int(k), sizes)
        rows.append({**{d: cats[d][i] for d, i in zip(ids, idx)}, "value": v})
    return pd.DataFrame(rows), cats


def eurostat(dataset, prefs, probe=None):
    """Fetch one series per (geo,time). Fail loud if filters leave >1 series.

    probe: extra filters for the metadata request only (large datasets return HTTP 413 without them).
    """
    base = f"{ES}/{dataset}"
    meta = _get(base, {"format": "JSON", "lang": "EN", "geo": "BG", "lastTimePeriod": 1, **(probe or {})})
    _, cats = _parse(meta) if meta.get("value") else (None, {d: list(meta["dimension"][d]["category"]["index"]) for d in meta["id"]})
    params = {"format": "JSON", "lang": "EN", "sinceTimePeriod": SINCE}
    for dim, cands in prefs.items():
        if dim in cats:
            hit = next((c for c in cands if c in cats[dim]), None)
            if hit:
                params[dim] = hit
            else:
                print(f"  [{dataset}] no candidate for '{dim}'. available: {cats[dim][:25]}")
    df, _ = _parse(_get(base, params))
    extra = [d for d in df.columns if d not in ("geo", "TIME_PERIOD", "time", "value") and df[d].nunique() > 1]
    if extra:
        raise RuntimeError(f"{dataset}: filters leave several series. Extra dims: "
                           f"{ {d: sorted(df[d].unique())[:15] for d in extra} }")
    df = df.rename(columns={"TIME_PERIOD": "year", "time": "year"})
    df["year"] = df["year"].astype(int)
    return df[["geo", "year", "value"]]


# ---------- World Bank ----------
def worldbank(ind):
    codes = ";".join(CT.iso2)
    rows, page = [], 1
    while True:
        js = _get(f"{WB}/country/{codes}/indicator/{ind}",
                  {"format": "json", "per_page": 5000, "date": f"{SINCE}:2025", "page": page})
        if len(js) < 2 or js[1] is None:
            break
        rows += [{"iso2": r["country"]["id"], "year": int(r["date"]), "value": r["value"]}
                 for r in js[1] if r["value"] is not None]
        if page >= js[0]["pages"]:
            break
        page += 1
    if not rows:
        raise RuntimeError(f"World Bank returned no data for {ind}")
    return pd.DataFrame(rows)


# ---------- build panel ----------
def latest(df, key="iso2"):
    d = df.sort_values("year").groupby(key).tail(1)
    return d[[key, "year", "value"]]


def build_real(fdi_mode):
    # Codes verified against the live API dimension metadata on 2026-09-30.
    income = eurostat("ilc_di03", {"statinfo": ["MED_EI"], "unit": ["PPS"],
                                   "age": ["TOTAL"], "sex": ["T"]})
    sat = eurostat("ilc_pw01", {"statinfo": ["AVG"], "unit": ["RTG"], "isced11": ["TOTAL"],
                                "life_sat": ["LIFE"], "age": ["Y_GE16"], "sex": ["T"]})
    for n, d in (("income", income), ("lifesat", sat)):
        d.to_csv(DATA / f"raw_{n}.csv", index=False)
        d["iso2"] = d.geo.map(ES2ISO)
    fdi, label = None, None
    if fdi_mode in ("auto", "eurostat"):
        try:
            # NI = "Net FDI inward" (stock). LIAB has almost no data. counterp has no TOTAL: use IMM.
            pos = eurostat("bop_fdi6_pos", {
                "currency": ["MIO_EUR"], "nace_r2": ["TOTAL"], "stk_flow": ["NI"], "entity": ["TOTAL"],
                "fdi_item": ["DI__D__F"], "counterp": ["IMM"], "partner": ["WRL_REST"]},
                probe={"freq": "A", "currency": "MIO_EUR", "nace_r2": "TOTAL"})
            gdp = eurostat("nama_10_gdp", {"unit": ["CP_MEUR"], "na_item": ["B1GQ"]})
            m = pos.merge(gdp, on=["geo", "year"], suffixes=("_pos", "_gdp"))
            m["value"] = m.value_pos / m.value_gdp * 100
            m["iso2"] = m.geo.map(ES2ISO)
            pos.to_csv(DATA / "raw_fdi_stock.csv", index=False)
            fdi, label = m.dropna(subset=["iso2"]), "Inward FDI stock, % of GDP (avg last 5y)"
        except Exception as e:  # noqa: BLE001 - any failure falls back to World Bank
            print(f"Eurostat FDI stock failed -> {e}")
            if fdi_mode == "eurostat":
                sys.exit(1)
            print("Falling back to World Bank FDI inflow proxy.")
    if fdi is None:
        fdi = worldbank("BX.KLT.DINV.WD.GD.ZS")
        fdi.to_csv(DATA / "raw_fdi_inflow_wb.csv", index=False)
        label = "FDI net inflow, % of GDP (avg last 5y) [flow proxy]"
    pop = worldbank("SP.POP.TOTL")
    pop.to_csv(DATA / "raw_pop.csv", index=False)
    return income, sat, fdi, pop, label


def build_demo():
    rng = np.random.default_rng(1)
    rows = []
    for _, r in CT.iterrows():
        east = r.group == "East"
        fdi = rng.uniform(20, 80) if east else rng.uniform(10, 60)
        if r.spe == "SPE hub":
            fdi = rng.uniform(300, 1500)
        inc = (14000 if east else 27000) + fdi * (20 if east else 8) + rng.normal(0, 1500)
        sat = (6.3 if east else 7.4) + rng.normal(0, .3)
        rows.append((r.iso2, fdi, inc, sat, rng.uniform(1e6, 8e7)))
    d = pd.DataFrame(rows, columns=["iso2", "fdi", "inc", "sat", "pop"])
    mk = lambda c: pd.DataFrame({"iso2": d.iso2, "year": 2023, "value": d[c]})
    fdi = pd.concat([mk("fdi").assign(year=y) for y in range(2019, 2024)])
    # make two countries stale so the demo exercises the grey path
    fdi = fdi[~((fdi.iso2 == "GB") & (fdi.year > 2019)) & ~((fdi.iso2 == "ME") & (fdi.year != 2020))]
    return mk("inc"), mk("sat"), fdi, mk("pop"), "SYNTHETIC DEMO FDI"


def make_panel(income, sat, fdi, pop):
    fdi = fdi[np.isfinite(fdi["value"])]  # NaN/inf (e.g. GDP 0) must not count as a data point
    if fdi.empty:
        raise RuntimeError("FDI series is empty after dropping missing values")
    latest_year = fdi.year.max()
    print(f"FDI feed newest year: {latest_year} (countries not reaching {FDI_MIN_POINTS} points since "
          f"{latest_year - FDI_YEARS + 1} are grey)")
    fdi = fdi.sort_values("year").groupby("iso2").tail(FDI_YEARS)
    fdi = fdi.assign(recent=fdi.year > latest_year - FDI_YEARS)
    f = fdi.groupby("iso2").agg(fdi=("value", "mean"), fdi_years=("year", "nunique"),
                                fdi_last_year=("year", "max"), n_recent=("recent", "sum")).reset_index()
    f["fdi_ok"] = f.n_recent >= FDI_MIN_POINTS
    p = CT.merge(f, on="iso2", how="left")
    p = p.merge(latest(income).rename(columns={"value": "income", "year": "income_year"}), on="iso2", how="left")
    p = p.merge(latest(sat).rename(columns={"value": "lifesat", "year": "lifesat_year"}), on="iso2", how="left")
    p = p.merge(latest(pop).rename(columns={"value": "pop"})[["iso2", "pop"]], on="iso2", how="left")
    p["pop"] = p["pop"].fillna(p["pop"].median())
    # bivariate classes: terciles from non-hub countries
    p["fdi_ok"] = p["fdi_ok"].fillna(False).astype(bool)
    core = p[(p.spe == "other") & p.fdi_ok]
    for col in ("fdi", "income", "lifesat"):
        q = core[col].quantile([1 / 3, 2 / 3]).values
        p[col + "_t"] = np.where(p[col].isna(), np.nan, np.digitize(p[col], q))
    p.loc[~p.fdi_ok, "fdi_t"] = np.nan
    for m in ("income", "lifesat"):
        ok = p["fdi_t"].notna() & p[m + "_t"].notna()
        p["cls_" + m] = np.where(ok, p["fdi_t"].fillna(0).astype(int).astype(str) + "_" +
                                 p[m + "_t"].fillna(0).astype(int).astype(str), None)
    return p


# ---------- charts ----------
# columns = FDI low->high, rows = quality low->high (Stevens 3x3 palette)
BV = {"0_0": "#e8e8e8", "1_0": "#dfb0d6", "2_0": "#be64ac",
      "0_1": "#ace4e4", "1_1": "#a5add3", "2_1": "#8c62aa",
      "0_2": "#5ac8c8", "1_2": "#5698b9", "2_2": "#3b4994"}


def legend_chart(name):
    d = pd.DataFrame([{"x": int(k[0]), "y": int(k[2]), "c": k} for k in BV])
    return (alt.Chart(d).mark_rect(stroke="white").encode(
        x=alt.X("x:O", title="FDI  low → high", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("y:O", title=f"{name}  low → high", sort="descending", axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("c:N", scale=alt.Scale(domain=list(BV), range=list(BV.values())), legend=None))
        .properties(width=120, height=120, title="Map key"))


def panel_chart(p, topo, ycol, yname, fdi_label, pick):
    d = p.copy()
    d["pop_m"] = (d["pop"] / 1e6).round(1)
    dd = d.dropna(subset=["fdi", ycol]).copy()
    dd["grp"] = np.where(dd.fdi_ok, dd.group, SPARSE)
    ycol_t = "cls_" + ycol
    dom, rng_ = ["West", "East", SPARSE], ["#1f77b4", "#d62728", "#9e9e9e"]
    tips = ["country", "group", "spe", alt.Tooltip("fdi:Q", format=".1f", title="FDI"),
            alt.Tooltip(f"{ycol}:Q", format=",.1f", title=yname), "pop_m",
            alt.Tooltip("fdi_last_year:Q", format="d", title="FDI latest year"),
            alt.Tooltip("fdi_years:Q", format="d", title="FDI years averaged")]
    yr = alt.Y(f"{ycol}:Q", title=yname, scale=alt.Scale(zero=False))
    xr = alt.X("fdi:Q", title=fdi_label, scale=alt.Scale(type="symlog", constant=50))
    base = alt.Chart(dd)
    pts = base.mark_circle(strokeWidth=2).encode(
        x=xr, y=yr,
        size=alt.Size("pop_m:Q", legend=None, scale=alt.Scale(range=[30, 500])),
        color=alt.Color("grp:N", scale=alt.Scale(domain=dom, range=rng_),
                        legend=alt.Legend(title="Bloc", orient="bottom", direction="horizontal")),
        stroke=alt.Stroke("spe:N", scale=alt.Scale(domain=["SPE hub", "other"], range=["black", "white"]),
                          legend=alt.Legend(title="Pass-through hub", orient="bottom", direction="horizontal")),
        opacity=alt.condition(pick, alt.value(0.9), alt.value(0.15)),
        tooltip=tips).add_params(pick)
    lab = base.mark_text(dy=-12, fontSize=10).encode(x=xr, y=yr, text="iso2:N",
                                                     opacity=alt.condition(pick, alt.value(1), alt.value(0.15)))
    reg = (alt.Chart(dd[(dd.spe == "other") & dd.fdi_ok])
           .transform_regression("fdi", ycol, groupby=["group"], method="linear")
           .mark_line(strokeDash=[4, 3]).encode(
               x=xr, y=yr,
               color=alt.Color("group:N", scale=alt.Scale(domain=dom, range=rng_), legend=None)))
    scatter = (pts + lab + reg).properties(width=460, height=H, title=f"{yname} vs FDI (regression excl. hubs and grey)")

    fields = ["country", "group", "spe", "fdi", ycol, ycol_t, "pop_m", "iso2", "fdi_last_year", "fdi_years"]
    shape = alt.Chart(topo).mark_geoshape().transform_lookup(
        lookup="id", from_=alt.LookupData(d, "iso_num", fields)
    ).encode(
        color=alt.condition(f"isValid(datum.{ycol_t})",
                            alt.Color(f"{ycol_t}:N", scale=alt.Scale(domain=list(BV), range=list(BV.values())),
                                      legend=None), alt.value("#f2f2f2")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.25)),
        # empty=False: nothing selected -> no outline (default would outline every country)
        stroke=alt.when(pick, empty=False).then(alt.value("black")).otherwise(alt.value("white")),
        strokeWidth=alt.when(pick, empty=False).then(alt.value(2.5)).otherwise(alt.value(0.5)),
        tooltip=[alt.Tooltip("country:N"), alt.Tooltip("fdi:Q", format=".1f"),
                 alt.Tooltip(f"{ycol}:Q", format=",.1f", title=yname),
                 alt.Tooltip("fdi_last_year:Q", format="d", title="FDI latest year"),
                 alt.Tooltip("fdi_years:Q", format="d", title="FDI years averaged")],
    ).project(type="mercator", center=[8, 54], scale=420, translate=[260, H / 2]).properties(width=520, height=H,
                                                                      title="Bivariate map (terciles)")
    short = {"income": "Income", "lifesat": "Life satisfaction"}[ycol]
    return alt.hconcat(scatter, shape, legend_chart(short)).resolve_scale(color="independent", stroke="independent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fdi", default="auto", choices=["auto", "eurostat", "wb"])
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    income, sat, fdi, pop, label = build_demo() if a.demo else build_real(a.fdi)
    p = make_panel(income, sat, fdi, pop)
    p.to_csv(DATA / ("panel_demo.csv" if a.demo else "panel.csv"), index=False)
    print(p[["country", "group", "fdi", "fdi_years", "fdi_last_year", "fdi_ok", "income", "income_year",
             "lifesat", "lifesat_year"]]
          .round(1).to_string(index=False))
    try:
        topo = alt.Data(values=requests.get(TOPO, timeout=60).json(),
                        format=alt.DataFormat(type="topojson", feature="countries"))
    except Exception as e:  # noqa: BLE001 - map then loads the topojson by URL
        print(f"Topojson download failed ({e}). Map in HTML loads it by URL instead.")
        topo = alt.topo_feature(TOPO, "countries")
    pick = alt.selection_point(fields=["iso2"], name="pick")
    top = panel_chart(p, topo, "income", "Median equiv. disposable income (PPS)", label, pick)
    pick2 = alt.selection_point(fields=["iso2"], name="pick2")
    bot = panel_chart(p, topo, "lifesat", "Life satisfaction (0-10)", label, pick2)
    note = ("Click a bubble to highlight it on the map. Shift+click for several. "
            "Bubble size = population. Black outline = pass-through FDI hub. "
            "Grey = FDI data stale or sparse. Hubs and grey countries are excluded from trend lines and tercile cuts.")
    chart = alt.vconcat(top, bot).properties(
        title=alt.TitleParams("FDI vs quality of life, Europe" + (" [SYNTHETIC DEMO]" if a.demo else ""),
                              subtitle=note, anchor="start", fontSize=18))
    chart.save(OUT / ("fdi_qol_demo.html" if a.demo else "fdi_qol.html"))
    print("Saved HTML.")


if __name__ == "__main__":
    main()
