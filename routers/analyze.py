import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
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

    stats     = run_statistical_analysis(df, numeric_cols)
    anomalies = detect_anomalies(df, numeric_cols)
    cat_stats = run_category_analysis(df, category_cols, numeric_cols)

    nlp_results = {}
    if text_cols:
        raw_nlp = run_sentiment_analysis(df, text_cols[0])
        nlp_results = {
            'distribution'  : raw_nlp['distribution'],
            'aspect_summary': raw_nlp['aspect_summary'],
            'top_keywords'  : [[kw, freq] for kw, freq in raw_nlp['top_keywords']],
            'records'       : raw_nlp['result_df'][['index','label','compound','pos','neu']].to_dict(orient='records')
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
        res = await client.post(f'https://api.telegram.org/bot{token}/sendMessage', json={
            'chat_id': payload.chat_id,
            'text': payload.message,
            'parse_mode': 'Markdown'
        })
        print(f"Telegram response: {res.status_code} {res.text}")
    return {'status': 'sent', 'telegram_response': res.text}

@router.post('/telegram-pdf')
async def send_telegram_pdf(
    chat_id: str = Form(...),
    message: str = Form(...),
    pdf: UploadFile = File(...)
):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        raise HTTPException(status_code=500, detail='Bot token not configured')
    async with httpx.AsyncClient(timeout=30) as client:
        # Kirim teks dulu
        await client.post(f'https://api.telegram.org/bot{token}/sendMessage', json={
            'chat_id': chat_id,
            'text': message
        })
        # Kirim PDF sebagai dokumen
        pdf_bytes = await pdf.read()
        res = await client.post(
            f'https://api.telegram.org/bot{token}/sendDocument',
            data={'chat_id': chat_id},
            files={'document': ('Report.pdf', pdf_bytes, 'application/pdf')}
        )
        print(f"Telegram PDF response: {res.status_code} {res.text}")
    return {'status': 'sent'}
