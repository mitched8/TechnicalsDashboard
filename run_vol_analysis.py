"""Headless vol-regime analysis: CSV in, report out. No UI required.

Usage:
    python run_vol_analysis.py --csv data/eurusd_daily_long.csv --name EURUSD
    python run_vol_analysis.py --demo            # synthetic data, labeled as such

Outputs (to --out dir, default ./vol_reports):
    <name>_report.json       headline state, cone, event studies, HAR eval, playbook
    <name>_states.csv        per-bar state + features
    <name>_events.csv        regime event log
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol_engine.data_io import load_ohlc_csv
from vol_engine.synthetic import generate_ohlc
from vol_engine.regime import classify_states, RegimeConfig, EventKind, VolState
from vol_engine.cones import vol_cone
from vol_engine.event_study import event_study, time_to_resolution
from vol_engine.har import har_forecast_report
from vol_engine.hmm import fit_hmm_2state
from vol_engine.playbook import get_playbook
from vol_engine.rv_estimators import log_returns


def analyze(df: pd.DataFrame, name: str) -> dict:
    result = classify_states(df)
    feats = result.features
    current_state = result.states.iloc[-1]

    report = {
        "name": name,
        "provenance": df.attrs.get("provenance", "unspecified"),
        "synthetic": bool(df.attrs.get("synthetic", False)),
        "bars": len(df),
        "date_range": [str(df.index.min().date()), str(df.index.max().date())],
        "current_state": current_state,
        "current_compression_score": _f(feats["compression_score"].iloc[-1]),
        "current_width_pctile": _f(feats["width_pctile"].iloc[-1]),
    }

    # Vol cone
    cone = vol_cone(df)
    report["vol_cone"] = cone.round(2).to_dict(orient="index") if not cone.empty else {}

    # Event studies for the two core signals
    studies = {}
    comp_mask = result.event_mask(EventKind.COMPRESSION_START)
    brk_mask = result.event_mask(EventKind.BREAK_CONFIRMED)
    for label, mask in [("compression_start", comp_mask), ("break_confirmed", brk_mask)]:
        es = event_study(df, mask, label)
        studies[label] = {
            "n_events": es.n_events_used,
            "warnings": es.warnings,
            "horizons": es.to_frame().to_dict(orient="records"),
        }
    report["event_studies"] = studies

    # Time-to-resolution for compression -> tenor mapping
    comp_idx = np.flatnonzero(comp_mask.values)
    resolution_mask = feats["expansion_bar"] | brk_mask
    res_stats = time_to_resolution(comp_idx, resolution_mask) if len(comp_idx) else {}
    report["compression_resolution"] = res_stats

    # HAR forecast evaluation (with regime augmentation)
    extra = pd.DataFrame({
        "compression_score": feats["compression_score"],
        "trend_dummy": (result.states == VolState.TREND.value).astype(float),
    })
    har = har_forecast_report(df, horizon=21, extra_features=extra)
    report["har_21d"] = {**har.summary(), "warnings": har.warnings}
    if har.forecast is not None and har.forecast.notna().any():
        report["har_21d"]["latest_forecast_vol_pct"] = _f(har.forecast.dropna().iloc[-1])

    # HMM cross-check
    hmm = fit_hmm_2state(log_returns(df["Close"]))
    if hmm is not None:
        report["hmm"] = {
            "p_high_vol_now": _f(hmm.smoothed_p_high.iloc[-1]),
            "sigma_low_ann_pct": _f(hmm.sigmas[0] * np.sqrt(252) * 100),
            "sigma_high_ann_pct": _f(hmm.sigmas[1] * np.sqrt(252) * 100),
            "converged": hmm.converged,
        }

    # Playbook
    recent_failed = False
    resolved = [e for e in result.events
                if e.kind in (EventKind.BREAK_CONFIRMED, EventKind.BREAK_FAILED)]
    if resolved:
        recent_failed = resolved[-1].kind == EventKind.BREAK_FAILED
    entry = get_playbook(current_state, res_stats or None, recent_failed)
    report["playbook"] = {
        "stance": entry.stance,
        "rationale": entry.rationale,
        "structures": entry.structures,
        "tenor_guidance": entry.tenor_guidance,
        "caveats": entry.caveats,
    }
    return report, result


def _f(x, nd=3):
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="OHLC CSV path (Date,Open,High,Low,Close)")
    ap.add_argument("--name", default=None, help="Instrument name for the report")
    ap.add_argument("--demo", action="store_true", help="Use labeled synthetic demo data")
    ap.add_argument("--out", default="vol_reports", help="Output directory")
    args = ap.parse_args()

    if args.demo:
        df, _ = generate_ohlc()
        name = args.name or "SYNTHETIC_DEMO"
    elif args.csv:
        df = load_ohlc_csv(args.csv)
        name = args.name or Path(args.csv).stem
    else:
        ap.error("Provide --csv PATH or --demo")

    report, result = analyze(df, name)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{name}_report.json").write_text(json.dumps(report, indent=2, default=str))
    states_frame = result.features.assign(vol_state=result.states)
    states_frame.to_csv(out_dir / f"{name}_states.csv")
    result.events_frame().to_csv(out_dir / f"{name}_events.csv", index=False)

    print(f"[{name}] {report['provenance']}")
    if report["synthetic"]:
        print("  *** SYNTHETIC DATA — engine demo only, never trade off this ***")
    print(f"  bars={report['bars']}  state={report['current_state']}  "
          f"compression_score={report['current_compression_score']}")
    print(f"  playbook: {report['playbook']['stance']}")
    for sig, es in report["event_studies"].items():
        print(f"  event study [{sig}]: n={es['n_events']}"
              + (f"  WARN: {es['warnings'][0]}" if es["warnings"] else ""))
    print(f"  HAR 21d: {report['har_21d']}")
    print(f"  reports written to {out_dir}/")


if __name__ == "__main__":
    main()
