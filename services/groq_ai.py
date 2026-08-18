import os
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

def build_prompt(profile: dict, stats: dict, anomalies: dict, nlp_results: dict) -> str:
    lines = []
    lines.append("Anda adalah Data Analyst Senior dengan spesialisasi Business Intelligence.")
    lines.append("Tugas Anda adalah memberikan analisis profesional berdasarkan data yang disediakan.")
    lines.append("Gunakan pendekatan statistik dalam menjelaskan temuan. Jangan gunakan kalimat template atau saran umum yang tidak berbasis angka.")
    lines.append("Rujuk langsung pada nilai mean, median, standar deviasi, dan distribusi data yang ada.")
    lines.append("")
    lines.append("**Format Output WAJIB mengikuti struktur ini (Bahasa Indonesia profesional):**")
    lines.append("")
    lines.append("**Ringkasan Kondisi Data**")
    lines.append("[tulis ringkasan datanya, sebutkan total baris/kolom, missing values, dan tipe data dominan]")
    lines.append("")
    lines.append("**Insight Statistik Utama**")
    lines.append("[jelaskan angka-angka penting dari statistik deskriptif. Misal: rata-rata penjualan, produk termahal, distribusi yang paling sering muncul]")
    lines.append("")
    lines.append("**Anomali & Pola Bisnis**")
    lines.append("[jelaskan pola aneh yang ditemukan (misal: ada lonjakan pembelian di jam tertentu, atau produk dengan harga tinggi tapi jarang laku)]")
    lines.append("")
    lines.append("**Kesimpulan & Rekomendasi Data-Driven**")
    lines.append("[berikan 3-5 poin rekomendasi yang KONKRET, spesifik, dan berdasarkan temuan statistik di atas]")
    lines.append("")
    lines.append("")

    # Masukkan data asli ke prompt
    lines.append(f"Total baris  : {profile['shape']['rows']}")
    lines.append(f"Total kolom  : {profile['shape']['columns']}")
    lines.append(f"Missing      : {profile.get('missing_values', 'tidak ada')}")
    lines.append("")

    if stats.get('descriptive'):
        lines.append("== STATISTIK DESKRIPTIF ==")
        for col, stat in stats['descriptive'].items():
            lines.append(f"{col}: {stat}")
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
            max_tokens=2000,
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
        'insight statistik utama': 'anomaly_flags', # agak dipaksa, tapi yg penting isinya
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
