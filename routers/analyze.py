import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.loader  import load_csv, detect_column_types, generate_data_profile
from services.stats   import run_statistical_analysis, detect_anomalies, run_category_analysis
from services.nlp     import run_sentiment_analysis
from services.groq_ai import build_prompt, call_groq, parse_groq_response

router = APIRouter()


@router.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='File harus berformat CSV')

    file_bytes = await file.read()

    df        = load_csv(file_bytes, file.filename)
    col_types = detect_column_types(df)
    profile   = generate_data_profile(df, col_types)

    numeric_cols  = profile['columns_by_type']['numeric']
    category_cols = profile['columns_by_type']['category']
    text_cols     = profile['columns_by_type']['text']

    stats      = run_statistical_analysis(df, numeric_cols)
    anomalies  = detect_anomalies(df, numeric_cols)
    cat_stats  = run_category_analysis(df, category_cols, numeric_cols)

    nlp_results = {}
    if text_cols:
        raw_nlp     = run_sentiment_analysis(df, text_cols[0])
        nlp_results = {
            'distribution'  : raw_nlp['distribution'],
            'aspect_summary': raw_nlp['aspect_summary'],
            'top_keywords'  : raw_nlp['top_keywords']
        }

    prompt  = build_prompt(profile, stats, anomalies, nlp_results)
    raw     = call_groq(prompt)
    insight = parse_groq_response(raw)

    return {
        'profile'  : profile,
        'stats'    : stats,
        'anomalies': anomalies,
        'sentiment': nlp_results,
        'insight'  : insight
    }
