# AI SubGen

Aplikasi localhost untuk membuat subtitle Bahasa Indonesia (.ass / .srt) — transkripsi Jepang via
Gemini 2.5 Pro, terjemahan via Gemini 2.5 Flash (atau GPT-4o / Claude).

## Persiapan

1. **Python 3.11+** dan **ffmpeg** harus terpasang:
   ```powershell
   winget install Gyan.FFmpeg   # lalu restart terminal
   ```
2. Install dependensi:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Salin `.env.example` menjadi `.env` lalu isi `GEMINI_API_KEY`
   (wajib). `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` opsional —
   provider tanpa key tidak muncul di dropdown.

## Menjalankan

```powershell
python -m uvicorn app.main:app --port 8000
```

Buka <http://127.0.0.1:8000>, pilih sumber (URL YouTube atau upload
mp3/m4a/wav/mp4/mkv), model translasi, dan format output, lalu klik
**Mulai**. Progress tampil live; setelah selesai muncul link unduhan
dan ringkasan biaya token (estimasi ~$0.20–$0.50 per episode).

## Output per job (`output/{job_id}/`)

| File | Isi |
|---|---|
| `result.ass` / `result.srt` | Subtitle final (style HikaLeon dari `context/template.ass`) |
| `transcript_jp.json` | Transkrip Jepang per ujaran (id, start, end, type, ja) |
| `translated_id.json` | Transkrip + terjemahan Indonesia |
| `flags.json` | Baris dengan CPS > 25 untuk dicek manual (tidak dipotong otomatis) |
| `usage.json` | Akumulasi token & estimasi biaya |
| `source.mp4` | Video sumber (hanya jika "Simpan video" dicentang) |

## Jika gagal di tengah jalan

Klik **Coba lagi** di UI. Setiap tahap (download, normalisasi, chunking,
transkripsi per chunk, translasi per batch) di-checkpoint ke disk —
retry melanjutkan dari tahap yang gagal dan **tidak pernah mengulang
transkripsi yang sudah selesai** (tahap paling mahal).

Jika download YouTube gagal karena yt-dlp usang:
```powershell
pip install -U yt-dlp
```

## Konteks terjemahan

`context/context.md` (style guide + glossary) dan `context/members.md`
(roster member) dibaca ulang **setiap job** — edit saja filenya, tidak
perlu restart server. `context/template.ass` menentukan style .ass;
tempel style dari episode HikaLeon asli untuk hasil yang identik.

## Menjalankan test

```powershell
python -m pytest -q
```
