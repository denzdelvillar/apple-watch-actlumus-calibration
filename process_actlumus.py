#!/usr/bin/env python3
"""Convert ActLumus DATE/TIME + LIGHT into a complete UTC time grid.

The input DATE/TIME is interpreted in the explicitly supplied local time zone.
Requested UTC range is [--start, --end). Empty intervals remain blank (NaN),
not zero or interpolated. Output: timestamp, lux, and a data-availability flag.
"""

import argparse
from pathlib import Path

import pandas as pd


def utc_iso_milliseconds(series: pd.Series) -> pd.Series:
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"


def parse_utc_bound(text: str, label: str) -> pd.Timestamp:
    if not text.endswith("Z"):
        raise ValueError(f"{label} must be an explicit UTC timestamp ending in Z: {text!r}")
    value = pd.to_datetime(text, utc=True, errors="raise").as_unit("ns")
    if pd.isna(value):
        raise ValueError(f"{label} must be a valid UTC timestamp")
    return value


def process_actlumus(
    input_csv: str,
    output_csv: str,
    timezone: str,
    start_time: str,
    end_time: str,
    resolution: str = "1s",
    conflict_policy: str = "error",
) -> pd.DataFrame:
    start = parse_utc_bound(start_time, "--start")
    end = parse_utc_bound(end_time, "--end")
    step = pd.to_timedelta(resolution)

    if start >= end:
        raise ValueError("--start must precede --end")
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("--resolution must be a positive fixed duration (e.g. 1s, 10s, 1min)")
    if conflict_policy not in {"error", "median"}:
        raise ValueError("--conflict-policy must be 'error' or 'median'")

    raw = pd.read_csv(input_csv)
    raw.columns = raw.columns.str.strip()
    for col in ("DATE/TIME", "LIGHT"):
        if col not in raw.columns:
            raise ValueError(f"Missing required column {col!r}. Available columns: {raw.columns.tolist()}")

    local_naive = pd.to_datetime(
        raw["DATE/TIME"].astype("string").str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    if local_naive.isna().any():
        examples = raw.loc[local_naive.isna(), "DATE/TIME"].head(5).tolist()
        raise ValueError(f"{local_naive.isna().sum()} unparseable DATE/TIME entries; examples: {examples}")

    try:
        timestamps = (
            local_naive.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
            .dt.tz_convert("UTC")
            .dt.as_unit("ns")
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot interpret DATE/TIME in timezone {timezone!r}. "
            "Check the zone and daylight-saving transitions."
        ) from exc

    lux = pd.to_numeric(raw["LIGHT"], errors="coerce")
    invalid_lux = lux.isna() & raw["LIGHT"].notna() & raw["LIGHT"].astype("string").str.strip().ne("")
    if invalid_lux.any():
        examples = raw.loc[invalid_lux, "LIGHT"].head(5).tolist()
        raise ValueError(f"{invalid_lux.sum()} nonnumeric LIGHT entries; examples: {examples}")

    data = pd.DataFrame({"timestamp": timestamps, "lux": lux})
    selected = data.loc[(data["timestamp"] >= start) & (data["timestamp"] < end)].copy()
    valid = selected.dropna(subset=["lux"])
    distinct = valid.drop_duplicates(subset=["timestamp", "lux"]).copy()

    unique_lux_count = distinct.groupby("timestamp")["lux"].nunique()
    conflicts = unique_lux_count[unique_lux_count > 1]
    if not conflicts.empty:
        conflict_rows = selected[selected["timestamp"].isin(conflicts.index)].copy()
        conflict_rows["timestamp"] = utc_iso_milliseconds(conflict_rows["timestamp"])
        report = Path(output_csv).with_name(Path(output_csv).stem + "_timestamp_conflicts.csv")
        report.parent.mkdir(parents=True, exist_ok=True)
        conflict_rows.to_csv(report, index=False)
        print(f"Conflicting same-timestamp ActLumus readings: {len(conflicts)}")
        print(f"Diagnostic file: {report}")
        if conflict_policy == "error":
            raise ValueError(
                "Conflicting LIGHT values at identical timestamps; inspect diagnostic CSV "
                "or explicitly use --conflict-policy median."
            )

    # Exactly one lux value per distinct timestamp; median only matters for conflicts.
    distinct = distinct.groupby("timestamp", as_index=False)["lux"].median()

    span = end - start
    number_of_intervals = span // step + (span % step != pd.Timedelta(0))
    number_of_intervals = int(number_of_intervals)
    grid = pd.date_range(start=start, periods=number_of_intervals, freq=step)

    # Anchor bins to user-supplied start. The final bin may be shorter than step;
    # --end is always an exclusive bound.
    if not distinct.empty:
        interval_id = ((distinct["timestamp"] - start) // step).astype("int64")
        means = distinct.groupby(interval_id)["lux"].mean()
    else:
        means = pd.Series(dtype="float64")

    output = pd.DataFrame({
        "timestamp": grid,
        "actlumus_lux": means.reindex(range(number_of_intervals)).to_numpy(),
    })
    output["has_actlumus_data"] = output["actlumus_lux"].notna()
    output["timestamp"] = utc_iso_milliseconds(output["timestamp"])

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Input rows: {len(raw):,}")
    print(f"Rows in requested range: {len(selected):,}")
    print(f"Distinct valid timestamp/lux pairs: {len(valid.drop_duplicates(['timestamp', 'lux'])):,}")
    print(f"Identical timestamp + LIGHT duplicates removed: {len(valid) - len(valid.drop_duplicates(['timestamp', 'lux'])):,}")
    print(f"Conflicting same-timestamp groups: {len(conflicts):,}")
    print(f"Input time zone: {timezone}")
    print(f"UTC range: [{start}, {end}) | resolution: {resolution}")
    print(f"Output intervals: {len(output):,}")
    print(f"Intervals containing LIGHT data: {output['actlumus_lux'].notna().sum():,}")
    print(f"Intervals with no LIGHT data: {output['actlumus_lux'].isna().sum():,}")
    print(f"Saved: {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Raw ActLumus CSV")
    parser.add_argument("--output", required=True, help="Processed CSV")
    parser.add_argument("--timezone", required=True,
                        help="Time zone of raw DATE/TIME, e.g. Asia/Singapore or Europe/Berlin")
    parser.add_argument("--start", required=True, help="Inclusive UTC start, ISO 8601 ending in Z")
    parser.add_argument("--end", required=True, help="Exclusive UTC end, ISO 8601 ending in Z")
    parser.add_argument("--resolution", default="1s", help="1s (default), 5s, 10s, 1min, etc.")
    parser.add_argument("--conflict-policy", choices=["error", "median"], default="error",
                        help="How to handle differing LIGHT at the exact same raw timestamp")
    args = parser.parse_args()
    process_actlumus(args.input, args.output, args.timezone, args.start,
                     args.end, args.resolution, args.conflict_policy)


if __name__ == "__main__":
    main()
