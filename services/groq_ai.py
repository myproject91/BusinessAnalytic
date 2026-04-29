import os
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))


def build_prompt(profile: dict, stats: dict, anomalies: dict, nlp_results: dict) -> str:
    lines = []
    lines.append("Kamu adalah analis bisnis senior dengan pengalaman 15 tahun.")
    lines.append("Format output WAJIB mengikuti struktur berikut secara ketat:")
    lines.append("")
    lines.append("**Ringkasan Kondisi Data**")
    lines.append("[tulis ringkasan di sini, maksimal 3 kalimat]")
    lines.append("")
    lines.append("**Anomali yang Ditemukan**")
    lines.append("[jelaskan anomali dan potensi masalah bisnisnya]")
    lines.append("")
    lines.append("**Pola Sentimen**")
    lines.append("[jelaskan pola sentimen dan artinya bagi bisnis]")
    lines.append("")
    lines.append("**Rekomendasi Bisnis**")
    lines.append("[tulis 3 sampai 5 poin rekomendasi konkret, gunakan format bernomor]")
    lines.append("")
    lines.append("**Strategi Improvement**")
    lines.append("[tulis strategi jangka pendek dan jangka panjang]")
    lines.append("")
    lines.append("Aturan: bahasa Indonesia, jangan tampilkan angka tanpa konteks, rekomendasi harus spesifik.")
    lines.append("")

    lines.append(f"Total baris  : {profile['shape']['rows']}")
    lines.append(f"Total kolom  : {profile['shape']['columns']}")
    lines.append(f"Missing      : {profile.get('missing_values', 'tidak ada')}")
    lines.append("")

    if stats.get('descriptive'):
        lines.append("== STATISTIK ==")
        for col, stat in stats['descriptive'].items():
            lines.append(f"{col}: {stat}")
        lines.append("")

    if anomalies:
        lines.append("== ANOMALI ==")
        for col, info in anomalies.items():
            lines.append(f"{col}: {info['count']} baris ({info['percent']}%) di luar batas [{info['lower_bound']} - {info['upper_bound']}]")
        lines.append("")

    if nlp_results:
        lines.append(f"== SENTIMENT == {nlp_results.get('distribution', {})}")
        if nlp_results.get('aspect_summary'):
            for aspect, counts in nlp_results['aspect_summary'].items():
                lines.append(f"  {aspect}: {counts}")
        lines.append("")

    return "\n".join(lines)


def call_groq(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {
                    'role'   : 'system',
                    'content': (
                        'Kamu adalah analis bisnis senior. '
                        'Ikuti format output yang diberikan secara ketat. '
                        'Tulis dalam bahasa Indonesia yang profesional.'
                    )
                },
                {'role': 'user', 'content': prompt}
            ],
            max_tokens  = 2000,
            temperature = 0.3,
            top_p       = 0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"error: {str(e)}"


def parse_groq_response(raw: str) -> dict:
    result = {'raw': raw, 'summary': '', 'anomaly_flags': '', 'sentiment': '', 'recommendations': '', 'strategy': ''}

    section_map = {
        'ringkasan kondisi data': 'summary',
        'anomali yang ditemukan': 'anomaly_flags',
        'pola sentimen'         : 'sentiment',
        'rekomendasi bisnis'    : 'recommendations',
        'strategi improvement'  : 'strategy'
    }

    current = 'summary'
    buffer  = []

    for line in raw.split('\n'):
        stripped   = line.strip()
        matched    = False

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
