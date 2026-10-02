"""Merge four preprocessed calibration CSVs on their common UTC timeline.

Requires pandas. Every input must have precisely the same set of timestamps;
misaligned inputs cause an informative error rather than silent data loss.
"""

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "actlumus": {"actlumus_lux", "has_actlumus_data"},
    "ambient": {"watch_mean_lux", "watch_sample_count", "has_watch_data"},
    "pedometer": {"pedometer_steps", "pedometer_update_count", "has_pedometer_data"},
    "diary": {"diary_location", "diary_on_wrist", "diary_is_moving", "has_diary_data"},
}


def load_source(name: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    required = {"timestamp"} | EXPECTED_COLUMNS[name]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing expected columns: {sorted(missing)}")

    # Parse and normalize datetime precision (some pandas versions infer us).
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .dt.as_unit("ns")
    )

    if df["timestamp"].isna().any():
        raise ValueError(f"{name}: {df['timestamp'].isna().sum()} invalid timestamps")

    duplicate_mask = df["timestamp"].duplicated(keep=False)
    if duplicate_mask.any():
        example = df.loc[duplicate_mask, "timestamp"].iloc[0]
        raise ValueError(
            f"{name}: {duplicate_mask.sum()} rows have duplicated timestamps "
            f"(example: {example}). Fix this before merging."
        )

    return df.sort_values("timestamp").reset_index(drop=True)


def merge_calibration(actlumus, ambient, pedometer, diary, output, resolution="1s"):
    step = pd.to_timedelta(resolution)
    if step <= pd.Timedelta(0):
        raise ValueError("Resolution must be positive")

    paths = {
        "actlumus": actlumus,
        "ambient": ambient,
        "pedometer": pedometer,
        "diary": diary,
    }
    sources = {name: load_source(name, path) for name, path in paths.items()}

    reference = sources["actlumus"]["timestamp"]
    if reference.empty:
        raise ValueError("ActLumus input is empty")
    if len(reference) > 1:
        gaps = reference.diff().dropna()
        if not gaps.eq(step).all():
            first_bad = gaps.loc[~gaps.eq(step)].index[0]
            raise ValueError(
                "ActLumus timeline is not uniformly spaced at "
                f"{resolution}: {reference.iloc[first_bad - 1]} -> "
                f"{reference.iloc[first_bad]}"
            )

    # All four processing scripts should use identical --start, --end,
    # --resolution. Check equality before any join to avoid silent truncation.
    for name, df in sources.items():
        if name == "actlumus":
            continue
        current = df["timestamp"]
        if not current.equals(reference):
            missing = reference[~reference.isin(current)].head(3)
            extra = current[~current.isin(reference)].head(3)
            raise ValueError(
                f"{name}: timestamps do not exactly match ActLumus. "
                f"Rows: ActLumus={len(reference):,}, {name}={len(current):,}. "
                f"Missing from {name}: {missing.astype(str).tolist()}. "
                f"Extra in {name}: {extra.astype(str).tolist()}. "
                "Check timezone, --start, --end, and --resolution."
            )

    # Fail on overlapping non-key column names instead of letting pandas
    # create ambiguous _x / _y suffixes.
    seen = set()
    for name, df in sources.items():
        columns = set(df.columns) - {"timestamp"}
        overlap = seen & columns
        if overlap:
            raise ValueError(f"{name}: overlapping data columns: {sorted(overlap)}")
        seen |= columns

    merged = sources["actlumus"]
    for name in ("ambient", "pedometer", "diary"):
        merged = merged.merge(
            sources[name], on="timestamp", how="left", validate="one_to_one"
        )

    # Both illuminance streams must have real lux values. Missing is not zero.
    merged["has_paired_light_data"] = (
        merged["actlumus_lux"].notna()
        & merged["watch_mean_lux"].notna()
    )

    merged["timestamp"] = (
        merged["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        .str[:-3] + "Z"
    )

    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(dest, index=False)

    print(f"Saved: {dest}")
    print(f"Timeline: {merged['timestamp'].iloc[0]} to {merged['timestamp'].iloc[-1]} (last interval start)")
    print(f"Resolution: {resolution}; rows: {len(merged):,}")
    print(f"Paired light intervals: {int(merged['has_paired_light_data'].sum()):,}")
    print(f"Diary-covered intervals: {int(merged['has_diary_data'].eq(True).sum()):,}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge ActLumus, Watch ambient light, pedometer, and diary CSVs."
    )
    parser.add_argument("--actlumus", required=True, help="actlumus_1hz.csv")
    parser.add_argument("--ambient", required=True, help="ambient_light_1hz.csv")
    parser.add_argument("--pedometer", required=True, help="pedometer_1hz.csv")
    parser.add_argument("--diary", required=True, help="diary_1hz.csv")
    parser.add_argument("--output", required=True, help="Merged CSV filename")
    parser.add_argument("--resolution", default="1s", help="Expected resolution, default 1s")
    args = parser.parse_args()
    merge_calibration(**vars(args))
