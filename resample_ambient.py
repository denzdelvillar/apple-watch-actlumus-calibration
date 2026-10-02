import argparse
from pathlib import Path

import pandas as pd


def format_utc(series):
    """Format UTC timestamps as ISO 8601 with millisecond precision."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"


def resample_ambient(
    input_csv,
    output_csv,
    start_time,
    end_time,
    resolution="1s",
    conflict_policy="error",
):
    # --------------------------------------------------
    # 1. Parse requested time range
    # --------------------------------------------------
    start = pd.to_datetime(start_time, utc=True).as_unit("ns")
    end = pd.to_datetime(end_time, utc=True).as_unit("ns")
    interval = pd.to_timedelta(resolution)

    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid start or end timestamp.")
    if start >= end:
        raise ValueError("Start time must be earlier than end time.")
    if interval <= pd.Timedelta(0):
        raise ValueError("Resolution must be greater than zero.")
    if conflict_policy not in {"error", "median"}:
        raise ValueError("conflict_policy must be 'error' or 'median'.")

    # The output timestamp is written to millisecond precision.
    # Reject finer precision rather than silently truncate it.
    millisecond = pd.Timedelta("1ms")
    if (
        start.value % millisecond.value != 0
        or end.value % millisecond.value != 0
        or interval.value % millisecond.value != 0
    ):
        raise ValueError(
            "Start, end, and resolution must be aligned to milliseconds "
            "because CSV timestamps are written with millisecond precision."
        )

    # --------------------------------------------------
    # 2. Load Apple Watch data
    # --------------------------------------------------
    df = pd.read_csv(input_csv)

    if not {"timestamp", "lux"}.issubset(df.columns):
        raise ValueError("Input CSV must contain timestamp and lux columns.")

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .dt.as_unit("ns")
    )
    df["lux"] = pd.to_numeric(df["lux"], errors="coerce")

    invalid_count = df[["timestamp", "lux"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["timestamp", "lux"]).copy()

    # --------------------------------------------------
    # 3. Filter to requested time range
    #    Start inclusive; end exclusive.
    # --------------------------------------------------
    df = df.loc[
        (df["timestamp"] >= start) & (df["timestamp"] < end)
    ].copy()
    raw_in_range = len(df)

    # --------------------------------------------------
    # 4. Resolve repeated timestamps BEFORE resampling
    # --------------------------------------------------
    # Each distinct timestamp must contribute only ONE lux value
    # to the mean, median, standard deviation, and sample count.
    timestamp_groups = df.groupby("timestamp")["lux"]
    group_sizes = timestamp_groups.size()
    distinct_lux_counts = timestamp_groups.nunique()

    repeated_groups = int((group_sizes > 1).sum())
    agreeing_groups = int(
        ((group_sizes > 1) & (distinct_lux_counts == 1)).sum()
    )
    conflicting_times = distinct_lux_counts.index[distinct_lux_counts > 1]
    conflicting_groups = len(conflicting_times)
    duplicate_rows_removed = raw_in_range - len(group_sizes)

    print(f"Input rows in requested range: {raw_in_range:,}")
    print(f"Rows dropped for invalid timestamp/lux (full CSV): {invalid_count:,}")
    print(f"Distinct timestamps: {len(group_sizes):,}")
    print(f"Repeated-timestamp groups: {repeated_groups:,}")
    print(f"  Agreeing lux values: {agreeing_groups:,}")
    print(f"  Conflicting lux values: {conflicting_groups:,}")
    print(f"Extra rows consolidated: {duplicate_rows_removed:,}")

    if conflicting_groups:
        output_path = Path(output_csv)
        report_path = output_path.with_name(
            output_path.stem + "_timestamp_conflicts.csv"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)

        conflicts = df.loc[
            df["timestamp"].isin(conflicting_times)
        ].sort_values("timestamp", kind="stable").copy()
        conflicts["timestamp"] = format_utc(conflicts["timestamp"])
        conflicts.to_csv(report_path, index=False)
        print(f"Conflicting records written to: {report_path}")

        if conflict_policy == "error":
            raise ValueError(
                f"Found {conflicting_groups:,} timestamps with different lux "
                "values. Inspect the conflict CSV, then rerun with "
                "--conflict-policy median only if you approve that rule."
            )
        print(
            "Using median lux for each conflicting timestamp; "
            "each timestamp still counts as ONE sample."
        )

    # When lux values agree, median leaves them unchanged.
    # When values conflict, median applies ONLY if explicitly allowed.
    unique_samples = (
        timestamp_groups.median()
        .rename("lux")
        .reset_index()
        .sort_values("timestamp")
    )

    # --------------------------------------------------
    # 5. Generate all output intervals
    # --------------------------------------------------
    edges = pd.date_range(start=start, end=end, freq=interval).as_unit("ns")
    if edges[-1] != end:
        edges = edges.append(pd.DatetimeIndex([end]))

    number_of_intervals = len(edges) - 1

    # --------------------------------------------------
    # 6. Assign unique measurements to intervals
    # --------------------------------------------------
    unique_samples["interval_id"] = pd.cut(
        unique_samples["timestamp"],
        bins=edges,
        right=False,
        labels=False,
    )

    # --------------------------------------------------
    # 7. Aggregate lux, counting UNIQUE timestamps
    # --------------------------------------------------
    aggregated = (
        unique_samples.groupby("interval_id")["lux"]
        .agg(
            watch_mean_lux="mean",
            watch_median_lux="median",
            watch_min_lux="min",
            watch_max_lux="max",
            watch_std_lux="std",
            watch_sample_count="count",
        )
        .reindex(range(number_of_intervals))
    )

    # No observation != zero lux.
    aggregated["watch_sample_count"] = (
        aggregated["watch_sample_count"].fillna(0).astype(int)
    )

    # --------------------------------------------------
    # 8. Construct output: ONE timestamp (interval start)
    # --------------------------------------------------
    output = aggregated.reset_index(drop=True)
    output.insert(0, "timestamp", edges[:-1])
    output["has_watch_data"] = output["watch_sample_count"] > 0
    output["timestamp"] = format_utc(output["timestamp"])

    # --------------------------------------------------
    # 9. Save CSV
    # --------------------------------------------------
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.6f")

    print(f"Output: {output_path}")
    print(f"Requested range: {start} to {end} (end exclusive)")
    print(f"Resolution: {resolution}")
    print(f"Total intervals: {len(output):,}")
    print(f"Intervals with Watch data: {int(output['has_watch_data'].sum()):,}")
    print(f"Intervals without Watch data: {int((~output['has_watch_data']).sum()):,}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Resample Apple Watch ambient-light data, counting each "
            "distinct timestamp only once."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--resolution", default="1s")
    parser.add_argument(
        "--conflict-policy",
        choices=["error", "median"],
        default="error",
        help=(
            "What to do when identical timestamps have different lux values "
            "(default: error; median: one median lux per timestamp)."
        ),
    )
    args = parser.parse_args()
    resample_ambient(
        input_csv=args.input,
        output_csv=args.output,
        start_time=args.start,
        end_time=args.end,
        resolution=args.resolution,
        conflict_policy=args.conflict_policy,
    )
