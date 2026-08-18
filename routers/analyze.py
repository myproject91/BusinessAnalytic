import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.loader import load_csv, detect_column_types, generate_data_profile
from services.stats import run_statistical_analysis, detect_anomalies, run_category_analysis
from services.nlp import run_sentiment_analysis
from services.groq_ai import build_prompt, call_groq, parse_groq_response

router = APIRouter()

@router.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    try:
        # 1. Validasi file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail='File harus berformat CSV')

        file_bytes = await file.read()

        # 2. Load data
        df = load_csv(file_bytes, file.filename)
        col_types = detect_column_types(df)
        profile = generate_data_profile(df, col_types)

        numeric_cols = profile['columns_by_type']['numeric']
        category_cols = profile['columns_by_type']['category']
        text_cols = profile['columns_by_type']['text']

        # 3. Analisis statistik dan anomali
        stats = run_statistical_analysis(df, numeric_cols)
        anomalies = detect_anomalies(df, numeric_cols)
        cat_stats = run_category_analysis(df, category_cols, numeric_cols)

        # 4. Analisis NLP / Sentimen (Dilindungi agar gak crash)
        nlp_results = {}
        if text_cols:
            try:
                raw_nlp = run_sentiment_analysis(df, text_cols[0])
                nlp_results = {
                    'distribution': raw_nlp['distribution'],
                    'aspect_summary': raw_nlp['aspect_summary'],
                    'top_keywords': [[kw, freq] for kw, freq in raw_nlp['top_keywords']],
                    'records': raw_nlp['result_df'][['index', 'label', 'compound', 'pos', 'neu']].to_dict(orient='records')
                }
            except Exception as nlp_error:
                # Jika NLP gagal, kasih tau error tapi server tetap jalan
                nlp_results = {'error': f"NLP analysis failed: {str(nlp_error)}"}

        # 5. Panggil Groq AI
        prompt = build_prompt(profile, stats, anomalies, nlp_results)
        raw = call_groq(prompt)

        # Cek apakah Groq mengembalikan error
        if raw.startswith('error:'):
            insight = {'error': raw}
        else:
            insight = parse_groq_response(raw)

        # 6. Return hasil
        return {
            'profile': profile,
            'stats': stats,
            'anomalies': anomalies,
            'sentiment': nlp_results,
            'insight': insight
        }

    except HTTPException:
        raise
    except Exception as e:
        # Tangkap error lain, kirim JSON error yang jelas ke frontend
        raise HTTPException(status_code=500, detail=f"ERROR DI BACKEND: {str(e)}")

import os, httpx
from pydantic import BaseModel

class TelegramPayload(BaseModel):
    chat_id: str
    message: str

@router.post('/telegram')
async def send_telegram(payload: TelegramPayload):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        raise HTTPException(status_code=500, detail='Bot token not configured')
    async with httpx.AsyncClient() as client:
        await client.post(f'https://api.telegram.org/bot{token}/sendMessage', json={
            'chat_id': payload.chat_id,
            'text': payload.message,
            'parse_mode': 'Markdown'
        })
    return {'status': 'sent'}
