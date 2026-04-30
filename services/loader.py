import chardet
import pandas as pd


def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    detected = chardet.detect(file_bytes)
    encoding = detected['encoding'] or 'utf-8'

    sample = file_bytes[:4096].decode(encoding, errors='replace')
    sep_candidates = {',': 0, ';': 0, '\t': 0, '|': 0}
    for sep in sep_candidates:
        sep_candidates[sep] = sample.count(sep)
    separator = max(sep_candidates, key=sep_candidates.get)

    import io
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=separator,
            encoding=encoding,
            encoding_errors='replace',
            on_bad_lines='warn'
        )
    except Exception:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', on_bad_lines='skip')

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
        .str.lower()
    )
    df = df.loc[:, ~df.columns.str.match(r'^unnamed')]
    df = df.dropna(how='all').reset_index(drop=True)

    # Force convert object-like columns to string
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'object':
            df[col] = df[col].astype(str)

    return df


def detect_column_types(df: pd.DataFrame) -> dict:
    col_types = {}
    n_rows = len(df)

    for col in df.columns:
        series   = df[col].dropna()
        n_unique = series.nunique()
        dtype    = df[col].dtype

        # Numeric
        if pd.api.types.is_numeric_dtype(dtype):
            col_types[col] = 'id' if n_unique == n_rows else 'numeric'
            continue

        # Datetime
        try:
            parsed     = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
            pct_parsed = parsed.notna().sum() / len(series)
            if pct_parsed >= 0.8:
                col_types[col] = 'datetime'
                continue
        except Exception:
            pass

        # String — pakai str dtype check lebih robust
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or dtype == 'object':
            avg_len = series.astype(str).str.len().mean()

            if avg_len >= 30:
                col_types[col] = 'text'
            elif n_unique == n_rows and avg_len < 30:
                col_types[col] = 'id'
            elif n_unique <= 20:
                col_types[col] = 'category'
            else:
                col_types[col] = 'category'
        else:
            # Fallback — cek avg length langsung
            try:
                avg_len = series.astype(str).str.len().mean()
                col_types[col] = 'text' if avg_len >= 30 else 'category'
            except Exception:
                col_types[col] = 'unknown'

    return col_types


def generate_data_profile(df: pd.DataFrame, col_types: dict) -> dict:
    profile = {}
    profile['shape'] = {'rows': int(df.shape[0]), 'columns': int(df.shape[1])}

    missing = df.isnull().sum()
    profile['missing_values'] = {
        col: {'count': int(missing[col]), 'percent': round(missing[col] / len(df) * 100, 2)}
        for col in df.columns if missing[col] > 0
    }

    profile['column_types']    = col_types
    profile['columns_by_type'] = {
        tipe: [col for col, t in col_types.items() if t == tipe]
        for tipe in ['numeric', 'datetime', 'category', 'text', 'id', 'unknown']
    }

    numeric_cols = profile['columns_by_type']['numeric']
    if numeric_cols:
        profile['numeric_stats'] = df[numeric_cols].describe().round(2).to_dict()

    category_cols = profile['columns_by_type']['category']
    if category_cols:
        profile['category_distribution'] = {
            col: df[col].value_counts().head(10).to_dict()
            for col in category_cols
        }

    text_cols = profile['columns_by_type']['text']
    if text_cols:
        profile['text_info'] = {
            col: {
                'avg_length': round(df[col].dropna().astype(str).str.len().mean(), 1),
                'max_length': int(df[col].dropna().astype(str).str.len().max()),
                'min_length': int(df[col].dropna().astype(str).str.len().min()),
            }
            for col in text_cols
        }

    return profile
