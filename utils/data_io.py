

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


_BASE_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _BASE_DIR.parent  # ali_c/
_PROJECT_ROOT = _PACKAGE_ROOT.parent  # repository root when running from source
_CANDIDATE_DATA_DIRS = [
    _PACKAGE_ROOT / "Data",   # ali_c/Data (installed package)
    _PROJECT_ROOT,   # repo-level Data
]
DEFAULT_DATA_DIR = next((p for p in _CANDIDATE_DATA_DIRS if p.is_dir()), _CANDIDATE_DATA_DIRS[-1])
DEFAULT_PROFILE_PATH = DEFAULT_DATA_DIR / "data_profile.json"


# -----------------------------
# Helpers reused by both stages
# -----------------------------
def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(t) for t in x]


def _sanitize_time_strings(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    st = s.astype(str).str.strip()
    st = st.str.replace(r"(?<=\d)T(?=\d)", " ", regex=True)   # ISO 'T' separator
    st = st.str.replace(r"\s*T$", "", regex=True)             # trailing 'T'
    st = st.str.replace(r"Z$", "+00:00", regex=True)          # 'Z' → UTC offset
    return st


def _to_datetime_strong(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    s_norm = _sanitize_time_strings(s)
    out = pd.to_datetime(s_norm, errors="coerce")
    if out.notna().any():
        return out

    # epoch heuristic
    s_num = pd.to_numeric(s_norm, errors="coerce")
    m = s_num.dropna().median() if s_num.notna().any() else None
    if m is not None:
        unit = None
        if 1e12 > m >= 1e9:
            unit = "s"
        elif 1e15 > m >= 1e12:
            unit = "ms"
        elif 1e19 > m >= 1e15:
            unit = "ns"
        if unit:
            out = pd.to_datetime(s_num, unit=unit, errors="coerce")
            if out.notna().any():
                return out

    # Excel serial days
    if m is not None and 20000 <= m <= 60000:
        return pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")

    return pd.to_datetime(s_norm, errors="coerce")


def _clean_numeric_series(sr: pd.Series) -> pd.Series:
    if sr.dtype == object:
        sr = sr.astype(str).str.replace(",", "", regex=False).str.strip()
        sr = sr.replace("", np.nan)
    return pd.to_numeric(sr, errors="coerce")


# =========================================
# Profile-driven Data loader
# =========================================
def load_data_profile(profile_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load the data profile JSON (array of datasets) from the Data folder.
    """
    path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Data profile not found at {path}. "
            f"Tried default Data directories: {_CANDIDATE_DATA_DIRS}"
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Data profile must be a list of dataset descriptors.")
    return data


def _as_int_or_none(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() == "none":
        return None
    try:
        return int(val)
    except Exception:
        return None


def load_profiled_dataset(
    tag: str,
    *,
    data_dir: Optional[Path] = None,
    profile_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read a dataset described in the Data/data_profile.json file.

    The profile entry should contain:
      - tag (string identifier)
      - filename (relative to Data/)
      - timestamp_column
      - reference_column (optional)
      - measurement_column (optional)
      - labels (list, optional)
      - tag_row (int)
      - data_row (int)
      - unit_label (str)
      - units_row (int or None)
      - description_row (int or None)

    Returns:
        (df, desc_dict) from `read_table_with_tags`.
    """
    profiles = load_data_profile(profile_path)
    match = next((p for p in profiles if str(p.get("tag")) == str(tag)), None)
    if not match:
        raise KeyError(f"No entry with tag '{tag}' found in data profile.")

    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    filename = match.get("filename")
    if not filename:
        raise ValueError(f"Profile '{tag}' is missing 'filename'.")
    file_path = root / filename

    tag_row = _as_int_or_none(match.get("tag_row"))
    data_row = _as_int_or_none(match.get("data_row"))
    units_row = _as_int_or_none(match.get("units_row"))
    desc_row = _as_int_or_none(match.get("description_row"))

    return read_table_with_tags(
        file_path=str(file_path),
        tags_row=tag_row if tag_row else 1,
        data_row=data_row if data_row else 2,
        description_row=desc_row,
        units_row=units_row,
    )


def load_dataframe_from_profile(
    profile: Dict[str, Any],
    *,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read a dataset described by a profile object and return a cleaned DataFrame plus descriptions.

    Ensures:
      - Respects optional description_row/units_row (may be None).
      - tag_row < data_row (data starts after tags).
      - Output columns are renamed to ['Timestamp', 'Reference', 'Measurement'].
      - Description mapping keys follow the renamed columns.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dictionary.")

    filename = profile.get("filename")
    if not filename:
        raise ValueError("Profile is missing 'filename'.")

    tag_row = _as_int_or_none(profile.get("tag_row")) or 1
    data_row = _as_int_or_none(profile.get("data_row")) or (tag_row + 1)
    units_row = _as_int_or_none(profile.get("units_row"))
    desc_row = _as_int_or_none(profile.get("description_row"))

    if data_row <= tag_row:
        raise ValueError(f"data_row ({data_row}) must be greater than tag_row ({tag_row}).")

    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    file_path = root / filename

    df_all, desc_dict = read_table_with_tags(
        file_path=str(file_path),
        tags_row=tag_row,
        data_row=data_row,
        description_row=desc_row,
        units_row=units_row,
    )

    t_col = profile.get("timestamp_column")
    ref_col = profile.get("reference_column")
    meas_col = profile.get("measurement_column")
    missing = [name for name in (t_col, ref_col, meas_col) if name not in df_all.columns]
    if missing:
        raise KeyError(f"Columns missing in data file {file_path}: {missing}")

    subset = df_all[[t_col, ref_col, meas_col]].copy()
    rename_map = {t_col: "Timestamp", ref_col: "Reference", meas_col: "Measurement"}
    subset = subset.rename(columns=rename_map)

    # Drop a potential header row accidentally read as data (values equal to column names)
    mask_header = (
        subset["Timestamp"].astype(str).str.strip().str.lower() == "timestamp"
    ) & (
        subset["Reference"].astype(str).str.strip().str.lower() == str(profile.get("reference_column", "")).strip().lower()
    )
    subset = subset.loc[~mask_header].reset_index(drop=True)

    def _desc(orig: str) -> str:
        return desc_dict.get(orig, str(orig))

    desc_out = {
        "Timestamp": _desc(t_col),
        "Reference": _desc(ref_col),
        "Measurement": _desc(meas_col),
    }
    return subset, desc_out


# =========================================
# 1) Read file using preamble row indices
# =========================================

def read_table_with_tags(
    file_path: str,
    *,
    tags_row: int,                 # 1-based; required
    data_row: int,                 # 1-based; required, must be >= 2
    description_row: Optional[int] = None,  # 1-based; optional
    units_row: Optional[int] = None,        # 1-based; optional
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Read CSV/XLS/XLSX/ODS using explicit header rows and replace column names with the tags row.

    Rows (1-based):
      - tags_row        (required): row containing the column tags to use as final headers
      - data_row        (required): first row of actual data; must be >= 2
      - description_row (optional): row containing human-readable descriptions
      - units_row       (optional): row containing units

    Header rule (same as `load_data_from_file`):
      - if data_row == 2  -> header=None (we will skip the tags row and apply our own headers)
      - if data_row > 2   -> header = data_row - 2  (0-based index for pandas)

    Returns:
        df       : full DataFrame with tags applied (columns aligned and deduplicated)
        tags     : the final list of tag strings applied as column names
        desc_dict: maps {tag: "Description (units)"}; if either part missing, falls back
                   to the available piece; if both missing, falls back to the tag.
    """
    # ---------- validations ----------
    if not file_path:
        raise ValueError("file_path is required.")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if tags_row is None:
        raise ValueError("tags_row is required.")
    if data_row is None or data_row < 2:
        raise ValueError("data_row is required and must be >= 2.")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    is_excel = ext in (".xlsx", ".xls")
    is_csv   = ext == ".csv"
    is_ods   = ext == ".ods"
    if not (is_excel or is_csv or is_ods):
        raise NotImplementedError(
            f"Unsupported extension {ext}. Use .csv, .xlsx/.xls, or .ods."
        )

    # ---------- small helpers ----------
    def _cell_to_text(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()

    def _dedupe(names: List[str]) -> List[str]:
        seen = {}
        out = []
        for n in names:
            key = n if n else "col"
            if key not in seen:
                seen[key] = 0
                out.append(key)
            else:
                seen[key] += 1
                out.append(f"{key}_{seen[key]}")
        return out

    # ---------- peek top rows to extract tags/desc/units ----------
    max_row_needed = max([r for r in [description_row, units_row, tags_row] if r is not None])
    read_kwargs = dict(header=None, nrows=max_row_needed)

    if is_csv:
        peek = pd.read_csv(file_path, **read_kwargs)
    elif is_excel:
        peek = pd.read_excel(file_path, **read_kwargs)
    else:  # .ods
        peek = pd.read_excel(file_path, engine="odf", **read_kwargs)

    tags_series  = peek.iloc[tags_row - 1]
    desc_series  = peek.iloc[description_row - 1] if description_row else None
    units_series = peek.iloc[units_row - 1] if units_row else None

    raw_tags = [(_cell_to_text(x) or f"col_{j+1}") for j, x in enumerate(tags_series.tolist())]
    tags = _dedupe(raw_tags)

    # ---------- read full data with correct header handling ----------
    header   = None if data_row == 2 else (data_row - 2)  # pandas is 0-based
    skiprows = (tags_row - 1) if data_row == 2 else None  # when header=None, skip the tags row

    if is_csv:
        df = pd.read_csv(file_path, header=header, skiprows=skiprows)
    elif is_excel:
        df = pd.read_excel(file_path, header=header, skiprows=skiprows)
    else:  # .ods
        df = pd.read_excel(file_path, engine="odf", header=header, skiprows=skiprows)

    # Align width and apply tags
    if len(tags) < df.shape[1]:
        # if there are more columns than tags, extend with generated names
        extra = [f"col_{i+1}" for i in range(len(tags), df.shape[1])]
        tags = _dedupe(tags + extra)
    elif len(tags) > df.shape[1]:
        # if there are fewer columns than tags, truncate
        tags = tags[: df.shape[1]]
    df.columns = tags

    # ---------- build description dict (same logic as in load_data_from_file) ----------
    desc_dict: Dict[str, str] = {}
    for j, tag in enumerate(tags):
        desc_txt = (
            _cell_to_text(desc_series.iloc[j])
            if isinstance(desc_series, pd.Series) and j < len(desc_series)
            else ""
        )
        unit_txt = (
            _cell_to_text(units_series.iloc[j])
            if isinstance(units_series, pd.Series) and j < len(units_series)
            else ""
        )
        if desc_txt and unit_txt:
            val = f"{desc_txt} ({unit_txt})"
        elif desc_txt or unit_txt:
            val = desc_txt if desc_txt else unit_txt
        else:
            val = tag
        desc_dict[tag] = val

    return df, desc_dict

# =========================================
# 2) Build {name: 2-column DataFrame} dict
# =========================================
def make_signal_frames(
    df: pd.DataFrame,
    *,
    time_column: str,
    lab_column: Optional[str] = None,
    analyzer_list: Union[str, Sequence[str], None] = None,
    inferential_list: Union[str, Sequence[str], None] = None,
) -> Dict[str, pd.DataFrame]:
    """
    From a wide table 'df', create a dict of tidy 2-column DataFrames:
      {"Lab": DataFrame[TimeStamp, Value],
       "<AnalyzerTag>": DataFrame[TimeStamp, Value],
       "<InferentialTag>": DataFrame[TimeStamp, Value], ...}

    Rules:
      - 'TimeStamp' column is parsed robustly from `time_column`.
      - Values are coerced to numeric where possible.
      - Rows with NaT in TimeStamp are dropped.
      - Sorted by TimeStamp and de-duplicated on TimeStamp (keep='last').

    Args:
        df: Input DataFrame already labeled by tags.
        time_column: Name of the time column in df.
        lab_column: Single Lab column name in df (optional).
        analyzer_list: One or many analyzer columns (str or sequence).
        inferential_list: One or many inferential columns (str or sequence).

    Returns:
        Dict[str, pd.DataFrame] mapping label -> tidy 2-col frame (TimeStamp, Value).
    """
    if time_column not in df.columns:
        raise ValueError(f"time_column '{time_column}' not found. Available: {list(df.columns)}")

    # Prepare common, cleaned time vector
    ts = _to_datetime_strong(df[time_column])
    valid_time_mask = ts.notna()

    def _make_one(col_name: str, out_key: str) -> Optional[pd.DataFrame]:
        if col_name not in df.columns:
            return None
        vals = _clean_numeric_series(df[col_name])
        out = pd.DataFrame({"TimeStamp": ts, "Value": vals})
        out = out[valid_time_mask].copy()
        # Optional: drop rows where Value is NaN (comment out if you want to keep them)
        out = out[out["Value"].notna()]
        out = out.sort_values("TimeStamp", kind="mergesort")
        out = out.drop_duplicates(subset="TimeStamp", keep="last").reset_index(drop=True)
        out.columns = ["TimeStamp", "Value"]  # enforce exact naming
        out.name = out_key  # convenience label
        return out

    result: Dict[str, pd.DataFrame] = {}

    # Lab (key must be exactly "Lab")
    if lab_column:
        lab_df = _make_one(lab_column, "Lab")
        if lab_df is not None:
            result["Lab"] = lab_df

    # Analyzers: use column names as keys (e.g., "Analyzer_QF2910")
    for col in _as_list(analyzer_list):
        df2 = _make_one(col, col)
        if df2 is not None:
            result[col] = df2

    # Inferentials: likewise, keys are the column names provided
    for col in _as_list(inferential_list):
        df2 = _make_one(col, col)
        if df2 is not None:
            result[col] = df2

    if not result:
        raise ValueError("No output frames were created. Check your column names.")
    return result
