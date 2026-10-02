#!/usr/bin/env python3
"""Convert a participant's interval diary to a regular UTC analysis timeline.

Intervals are [start, end). The single output timestamp is the *start* of
an output bin. For non-1s resolutions, diary context is evaluated at that
instant; it is not a claim that context held throughout the whole bin.

No gaps are filled with inferred diary context. Zero-duration entries are
reported and omitted; overlapping positive-duration entries cause an error.
"""

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
REQUIRED_COLUMNS = ("start", "end", "location", "on_wrist", "is_moving", "notes")


def format_utc(series: pd.Series) -> pd.Series:
    """UTC ISO-8601, milliseconds (matching the other processed files)."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"


def parse_diary_boolean(values: pd.Series, column: str) -> pd.Series:
    """Parse true/false, leaving blanks missing rather than falsely setting False."""
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "": pd.NA})
    bad = normalized.notna() & ~normalized.isin(["true", "false", ""])
    if bad.any():
        lines = [int(i) + 2 for i in values.index[bad]]
        examples = values.loc[bad].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Invalid {column} value(s) at CSV line(s) {lines[:20]}: {examples}. "
            "Use true, false, or a blank."
        )
    return mapped.astype("boolean")


def process_diary(
    input_csv: str,
    output_csv: str,
    timezone: str,
    start_time: str,
    end_time: str,
    resolution: str = "1s",
) -> pd.DataFrame:
    # Requested analysis range is always UTC/offset-aware, not a naive local time.
    start = pd.to_datetime(start_time, utc=True, errors="raise").as_unit("ns")
    end = pd.to_datetime(end_time, utc=True, errors="raise").as_unit("ns")
    step = pd.to_timedelta(resolution)
    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("--start must be earlier than --end.")
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("--resolution must be a positive fixed duration (e.g. 1s, 10s, 1min).")
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {timezone!r}") from exc

    df = pd.read_csv(input_csv, dtype="string", keep_default_na=False)
    df.columns = df.columns.str.strip()
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    df = df.copy()
    df["_source_line"] = df.index + 2  # Header is line 1.
    df["diary_interval_id"] = [f"D{i:03d}" for i in range(1, len(df) + 1)]

    for field in ("start", "end"):
        local_naive = pd.to_datetime(
            df[field].str.strip(), format=DATE_FORMAT, errors="coerce"
        )
        bad = local_naive.isna()
        if bad.any():
            raise ValueError(
                f"Invalid or blank {field} at CSV line(s) "
                f"{df.loc[bad, '_source_line'].tolist()}. "
                f"Expected DD/MM/YYYY HH:MM:SS."
            )
        try:
            df[f"_{field}_utc"] = (
                local_naive.dt.tz_localize(
                    tz, ambiguous="raise", nonexistent="raise"
                )
                .dt.tz_convert("UTC")
                .dt.as_unit("ns")
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Cannot localize diary {field} with timezone {timezone}: {exc}. "
                "Check ambiguous/nonexistent times at DST transitions."
            ) from exc

    df["diary_on_wrist"] = parse_diary_boolean(df["on_wrist"], "on_wrist")
    df["diary_is_moving"] = parse_diary_boolean(df["is_moving"], "is_moving")
    df["diary_location"] = df["location"].str.strip().replace("", pd.NA)
    df["diary_notes"] = df["notes"].str.strip().replace("", pd.NA)

    backwards = df["_end_utc"] < df["_start_utc"]
    if backwards.any():
        raise ValueError(
            "Diary end is earlier than start on CSV line(s) "
            f"{df.loc[backwards, '_source_line'].tolist()}."
        )
    zero = df["_end_utc"] == df["_start_utc"]
    if zero.any():
        print(
            "Warning: skipping zero-duration diary interval(s) on CSV line(s): "
            + ", ".join(map(str, df.loc[zero, "_source_line"]))
        )
    valid = df.loc[~zero].sort_values(
        ["_start_utc", "_end_utc"], kind="stable"
    ).reset_index(drop=True)

    # Adjacent intervals sharing a boundary are fine; genuine overlaps are not.
    if len(valid) > 1:
        overlaps = valid["_start_utc"].iloc[1:].reset_index(drop=True) < (
            valid["_end_utc"].iloc[:-1].reset_index(drop=True)
        )
        if overlaps.any():
            row_indices = overlaps[overlaps].index.tolist()
            pairs = [
                (int(valid.loc[i, "_source_line"]), int(valid.loc[i + 1, "_source_line"]))
                for i in row_indices[:20]
            ]
            raise ValueError(
                f"Overlapping diary intervals at CSV line pairs {pairs}. "
                "Resolve them instead of arbitrarily choosing a diary state."
            )

    # Include partial final bin: its start must still be strictly < requested end.
    grid = pd.date_range(start=start, end=end, freq=step, inclusive="left").as_unit("ns")
    output = pd.DataFrame({"timestamp": grid})
    for field in (
        "diary_interval_id", "diary_location", "diary_on_wrist",
        "diary_is_moving", "diary_notes",
    ):
        output[field] = pd.NA
    output["has_diary_data"] = False

    if not valid.empty and not output.empty:
        # A diary interval applies only when start <= grid timestamp < end.
        # This assigns context by time without materializing individual diary seconds.
        starts = pd.DatetimeIndex(valid["_start_utc"])
        ends = pd.DatetimeIndex(valid["_end_utc"])
        locations = starts.searchsorted(grid, side="right") - 1
        candidate = locations >= 0
        is_covered = candidate.copy()
        is_covered[candidate] = (
            grid[candidate] < ends.take(locations[candidate])
        )
        target_rows = output.index[is_covered]
        source_rows = valid.iloc[locations[is_covered]]
        for col in (
            "diary_interval_id", "diary_location", "diary_on_wrist",
            "diary_is_moving", "diary_notes",
        ):
            output.loc[target_rows, col] = source_rows[col].to_numpy()
        output.loc[target_rows, "has_diary_data"] = True

    output["diary_on_wrist"] = output["diary_on_wrist"].astype("boolean")
    output["diary_is_moving"] = output["diary_is_moving"].astype("boolean")
    output["timestamp"] = format_utc(output["timestamp"])
    output = output[[
        "timestamp", "diary_interval_id", "diary_location",
        "diary_on_wrist", "diary_is_moving", "diary_notes", "has_diary_data",
    ]]

    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Output: {path}")
    print(f"Diary timezone: {timezone}")
    print(f"UTC range: [{start}, {end})")
    print(f"Resolution: {resolution}")
    print(f"Input diary intervals: {len(df):,}; zero-duration skipped: {int(zero.sum()):,}")
    print(f"Output rows: {len(output):,}")
    print(f"Rows with diary context: {int(output['has_diary_data'].sum()):,}")
    print(f"Rows without diary context: {int((~output['has_diary_data']).sum()):,}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Map local-time participant diary intervals onto a regular UTC timeline."
    )
    parser.add_argument("--input", required=True, help="Raw diary CSV")
    parser.add_argument("--output", required=True, help="Processed diary CSV")
    parser.add_argument(
        "--timezone", required=True,
        help="IANA timezone of diary entries, e.g. Asia/Singapore or Europe/Berlin",
    )
    parser.add_argument("--start", required=True, help="UTC/offset-aware ISO start (inclusive)")
    parser.add_argument("--end", required=True, help="UTC/offset-aware ISO end (exclusive)")
    parser.add_argument("--resolution", default="1s", help="Fixed interval width, default 1s")
    args = parser.parse_args()
    process_diary(args.input, args.output, args.timezone, args.start, args.end, args.resolution)


if __name__ == "__main__":
    main()
