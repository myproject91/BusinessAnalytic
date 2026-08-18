import os
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

def build_prompt(profile: dict, stats: dict, anomalies: dict, nlp_results: dict) -> str:
    lines = []
    lines.append("Anda adalah Data Analyst Senior dengan spesialisasi Business Intelligence.")
    lines.append("Tugas Anda adalah memberikan analisis profesional berdasarkan data yang disediakan.")
    lines.append("Gunakan pendekatan statistik dalam menjelaskan temuan. Jangan gunakan kalimat template.")
    lines.append("Rujuk langsung pada nilai mean, median, dan distribusi data yang ada.")
    lines.append("")
    lines.append("**Format Output WAJIB mengikuti struktur ini (Bahasa Indonesia profesional):**")
    lines.append("")
    lines.append("**Ringkasan Kondisi Data**")
    lines.append("[tulis ringkasan, sebutkan total baris/kolom, missing values, tipe data dominan]")
    lines.append("")
    lines.append("**Insight Statistik Utama**")
    lines.append("[jelaskan angka rata-rata, produk/harga tertinggi/terendah, dan distribusi kategori yang dominan]")
    lines.append("")
    lines.append("**Anomali & Pola Bisnis**")
    lines.append("[jelaskan anomali yang ditemukan beserta dampak bisnisnya]")
    lines.append("")
    lines.append("**Kesimpulan & Rekomendasi Data-Driven**")
    lines.append("[berikan 3-5 poin rekomendasi KONKRET berdasarkan temuan statistik di atas]")
    lines.append("")
    lines.append("")

    # Hanya kirim data esensial, jangan kirim tabel korelasi atau distribusi detail
    lines.append(f"Total baris  : {profile['shape']['rows']}")
    lines.append(f"Total kolom  : {profile['shape']['columns']}")
    lines.append(f"Missing      : {profile.get('missing_values', 'tidak ada')}")
    lines.append("")

    if stats.get('descriptive'):
        lines.append("== STATISTIK DESKRIPTIF (Mean / Median) ==")
        for col, stat in stats['descriptive'].items():
            lines.append(f"{col}: mean={stat['mean']}, median={stat['50%']}, min={stat['min']}, max={stat['max']}")
        lines.append("")

    if anomalies:
        lines.append("== DETEKSI ANOMALI ==")
        for col, info in anomalies.items():
            lines.append(f"{col}: {info['count']} baris ({info['percent']}%) di luar batas normal [{info['lower_bound']} - {info['upper_bound']}]")
        lines.append("")

    if nlp_results and nlp_results.get('distribution'):
        lines.append(f"== ANALISIS SENTIMEN PELANGGAN ==")
        lines.append(f"Distribusi sentimen: {nlp_results.get('distribution', {})}")
        if nlp_results.get('aspect_summary'):
            for aspect, counts in nlp_results['aspect_summary'].items():
                lines.append(f"  Aspek {aspect}: {counts}")
        lines.append("")

    return "\n".join(lines)

def call_groq(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model='groq/compound',
            messages=[
                {
                    'role': 'system',
                    'content': 'Anda adalah Data Analyst profesional. Analisis data dengan tajam menggunakan statistik. Berikan wawasan bisnis yang konkret, bukan saran umum.'
                },
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=1000,  # Turunkan dari 2000 ke 1000 agar response lebih ringkas
            temperature=0.2,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"error: {str(e)}"

def parse_groq_response(raw: str) -> dict:
    result = {'raw': raw, 'summary': '', 'anomaly_flags': '', 'sentiment': '', 'recommendations': '', 'strategy': ''}
    section_map = {
        'ringkasan kondisi data': 'summary',
        'insight statistik utama': 'anomaly_flags',
        'anomali & pola bisnis': 'sentiment',
        'kesimpulan & rekomendasi': 'recommendations'
    }
    current = 'summary'
    buffer = []
    for line in raw.split('\n'):
        stripped = line.strip()
        matched = False
        if stripped.startswith('**') and stripped.endswith('**'):
            heading = stripped.replace('**', '').lower().strip()
            for keyword, section in section_map.items():
                if keyword in heading:
                    if buffer:
                        result[current] += '\n'.join(buffer).strip() + '\n'
                        buffer = []
                    current = section
                    matched = True
                    break
        if not matched:
            buffer.append(line)
    if buffer:
        result[current] += '\n'.join(buffer).strip()
    for key in result:
        if key != 'raw':
            result[key] = result[key].strip()
    return result
