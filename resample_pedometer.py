#!/usr/bin/env python3
"""Create a regular, interval-end snapshot of cumulative Apple Watch pedometer data.

The raw end_date is the measurement-time reference; raw timestamp is NOT used
for alignment. Output has only one timestamp column (the interval start).

Intervals are [start, start + resolution), ... [last_start, end).
Each row's cumulative counter is the last observed state strictly before the
interval end, provided it is no older than --max-age at that interval end.
This is an offline snapshot, not an estimate of per-second steps or distance.
"""

import argparse
from pathlib import Path

import pandas as pd


def iso_utc_ms(times: pd.Series) -> pd.Series:
    """UTC timestamps with millisecond precision, e.g. 2026-09-13T05:11:03.000Z."""
    return times.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"


def resample_pedometer(
    input_csv: str,
    output_csv: str,
    start_time: str,
    end_time: str,
    resolution: str = "1s",
    max_age: str = "10s",
) -> pd.DataFrame:
    start = pd.to_datetime(start_time, utc=True)
    end = pd.to_datetime(end_time, utc=True)
    step = pd.to_timedelta(resolution)
    age_limit = pd.to_timedelta(max_age)

    if pd.isna(start) or pd.isna(end) or start >= end:
        raise ValueError("Provide a valid range with --start earlier than --end.")
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("--resolution must be a positive fixed duration, e.g. 1s, 10s, 1min.")
    if pd.isna(age_limit) or age_limit < pd.Timedelta(0):
        raise ValueError("--max-age must be a nonnegative duration, e.g. 10s.")

    df = pd.read_csv(input_csv)
    required = {"timestamp", "steps", "distance_metres", "start_date", "end_date"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing input CSV columns: {sorted(missing)}")

    # Do not discard repeated 'timestamp' values: they describe distinct end_dates.
    # Parse all fields first so equivalent representations deduplicate correctly.
    for col in ("timestamp", "start_date", "end_date"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.as_unit("ns")
    for col in ("steps", "distance_metres"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid = df[list(required)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"Found {invalid.sum()} rows with missing/invalid required data. "
            "Inspect the raw input before processing."
        )
    if (df["steps"] % 1 != 0).any():
        raise ValueError("Expected integer cumulative step counts; found fractional values.")

    original_count = len(df)
    df = df.drop_duplicates(subset=[
        "timestamp", "steps", "distance_metres", "start_date", "end_date"
    ]).copy()
    exact_duplicates = original_count - len(df)

    # end_date was unique in the examined dataset. Verify that assumption on
    # each new file rather than silently selecting an arbitrary conflicting state.
    end_groups = df.groupby("end_date", sort=False)
    bad_step = end_groups["steps"].nunique(dropna=False) > 1
    bad_distance = end_groups["distance_metres"].nunique(dropna=False) > 1
    bad_end_dates = bad_step.index[bad_step | bad_distance]
    if len(bad_end_dates):
        examples = [str(x) for x in bad_end_dates[:3]]
        raise ValueError(
            f"Found {len(bad_end_dates)} end_date values with conflicting "
            f"counters (examples: {examples}). Inspect raw records first."
        )
    # If the same end_date is repeated with agreeing counters, treat it as
    # one measurement for the 1 Hz analysis, retaining raw records separately.
    states = (
        df.sort_values(["end_date", "timestamp"], kind="stable")
          .drop_duplicates(subset=["end_date"], keep="first")
          .loc[:, ["end_date", "steps", "distance_metres"]]
          .sort_values("end_date")
          .rename(columns={
              "end_date": "_measured_at",
              "steps": "pedometer_steps",
              "distance_metres": "pedometer_distance_metres",
          })
    )

    # Anchor bins to the exact requested --start, not to the wall-clock minute.
    boundaries = pd.date_range(start=start, end=end, freq=step).as_unit("ns")
    if boundaries[-1] != end:
        boundaries = boundaries.append(pd.DatetimeIndex([end]))

    output = pd.DataFrame({
        "timestamp": boundaries[:-1],
        "_bin_end": boundaries[1:],
    })
    # Samples at exactly the requested end are excluded; a sample at a bin's
    # right boundary belongs to the NEXT bin, never the previous bin.
    output["_asof"] = output["_bin_end"] - pd.Timedelta(1, unit="ns")

    in_range = states.loc[
        (states["_measured_at"] >= start) & (states["_measured_at"] < end),
        ["_measured_at"],
    ].copy()
    in_range["_bin"] = pd.cut(
        in_range["_measured_at"], bins=boundaries, right=False, labels=False
    )
    counts = in_range.groupby("_bin").size().reindex(
        range(len(output)), fill_value=0
    )
    output["pedometer_update_count"] = counts.to_numpy(dtype="int64")

    # Explicitly match datetime precision for pandas versions that may infer
    # microseconds for raw CSV dates and nanoseconds for generated boundaries.
    states["_measured_at"] = states["_measured_at"].dt.as_unit("ns")
    output["_asof"] = output["_asof"].dt.as_unit("ns")

    # Include earlier measurements so the first output interval can inherit a
    # recent state. Hindsight-based: state is assigned at the END of each bin.
    matched = pd.merge_asof(
        output.sort_values("_asof"),
        states,
        left_on="_asof",
        right_on="_measured_at",
        direction="backward",
    )
    age = matched["_bin_end"] - matched["_measured_at"]
    fresh = matched["_measured_at"].notna() & (age <= age_limit)

    matched["pedometer_age_seconds"] = age.dt.total_seconds().where(fresh)
    matched["pedometer_steps"] = (
        matched["pedometer_steps"].where(fresh).astype("Int64")
    )
    matched["pedometer_distance_metres"] = (
        matched["pedometer_distance_metres"].where(fresh)
    )
    matched["has_pedometer_data"] = fresh

    result = matched.loc[:, [
        "timestamp",
        "pedometer_steps",
        "pedometer_distance_metres",
        "pedometer_update_count",
        "pedometer_age_seconds",
        "has_pedometer_data",
    ]].copy()
    result["timestamp"] = iso_utc_ms(result["timestamp"])

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False, float_format="%.9f")

    print(f"Output: {destination}")
    print(f"Input rows: {original_count:,}; exact duplicates removed: {exact_duplicates:,}")
    print(f"Distinct end_date measurements: {len(states):,}")
    print(f"Range: [{start}, {end}); resolution: {resolution}; max age: {max_age}")
    print(f"Output intervals: {len(result):,}")
    print(f"Intervals with actual updates: {(result['pedometer_update_count'] > 0).sum():,}")
    print(f"Intervals with fresh state: {result['has_pedometer_data'].sum():,}")
    print(f"Intervals without fresh state: {(~result['has_pedometer_data']).sum():,}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample raw cumulative pedometer snapshots using end_date."
    )
    parser.add_argument("--input", required=True, help="Raw pedometer CSV")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--start", required=True, help="Inclusive ISO 8601 UTC time")
    parser.add_argument("--end", required=True, help="Exclusive ISO 8601 UTC time")
    parser.add_argument("--resolution", default="1s", help="e.g. 1s, 10s, 1min")
    parser.add_argument("--max-age", default="10s", help="e.g. 10s, 5s; default 10s")
    args = parser.parse_args()
    resample_pedometer(
        args.input, args.output, args.start, args.end, args.resolution, args.max_age
    )
