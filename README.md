# SubtitledByAI

Aplikasi localhost untuk membuat subtitle Bahasa Indonesia (.ass / .srt) dari video variety show idol Jepang — transkripsi via Gemini 2.5 Pro, terjemahan via Gemini 2.5 Flash (atau GPT-4o / Claude).

Mendukung tiga grup dengan pilihan show masing-masing:

| Grup | Show |
|---|---|
| Nogizaka46 | Nogizaka Kojichuu, Haishinchuu, Enchouchuu |
| Hinatazaka46 | Hinatazaka de Aimashou, Narimashou, Channel |
| Sakurazaka46 | Channel, Chokosaku, Sokomagattara |

---

## Cara menjalankan

### Opsi A — Docker (direkomendasikan)

Tidak perlu install Python atau ffmpeg secara manual.

**Prasyarat:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) sudah terinstal dan berjalan.

1. Salin `.env.example` menjadi `.env` lalu isi `GEMINI_API_KEY`.
2. Build image (hanya perlu sekali, atau setelah update kode):
   ```powershell
   docker compose build
   ```
3. Jalankan:
   ```powershell
   docker compose up
   ```
4. Buka <http://localhost:8000>.

Untuk menghentikan: `docker compose down`.

> **Catatan volumes:** `output/` (subtitle hasil), `context/` (roster & glossary), dan `cookie.txt` di-mount dari folder lokal — aman di-edit kapan saja tanpa rebuild, dan tidak hilang saat container dimatikan.

---

### Opsi B — Lokal (tanpa Docker)

**Prasyarat:** Python 3.11+ dan ffmpeg harus terpasang:
```powershell
winget install Gyan.FFmpeg   # lalu restart terminal
```

1. Buat virtual environment dan install dependensi:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Salin `.env.example` menjadi `.env` lalu isi `GEMINI_API_KEY`.
3. Jalankan server:
   ```powershell
   python -m uvicorn app.main:app --port 8000
   ```
4. Buka <http://127.0.0.1:8000>.

---

## Penggunaan

1. Pilih **Grup** (Nogizaka46 / Hinatazaka46 / Sakurazaka46).
2. Pilih **Show** — daftar otomatis menyesuaikan grup yang dipilih.
3. Pilih sumber: URL YouTube atau upload file (mp3/m4a/wav/mp4/mkv).
4. Pilih model terjemahan dan format output.
5. Klik **Mulai**. Progress tampil live; setelah selesai muncul link unduhan dan estimasi biaya token (~$0.20–$0.50 per episode).

`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` bersifat opsional — provider tanpa key tidak muncul di dropdown.

---

## Output per job (`output/{job_id}/`)

| File | Isi |
|---|---|
| `result.ass` / `result.srt` | Subtitle final |
| `transcript_jp.json` | Transkrip Jepang per ujaran |
| `translated_id.json` | Transkrip + terjemahan Indonesia |
| `flags.json` | Baris dengan CPS > 25 untuk dicek manual |
| `usage.json` | Akumulasi token & estimasi biaya |
| `source.mp4` | Video sumber (hanya jika "Simpan video" dicentang) |

---

## Jika gagal di tengah jalan

Klik **Coba lagi** di UI. Setiap tahap (download, normalisasi, chunking, transkripsi per chunk, translasi per batch) di-checkpoint ke disk — retry melanjutkan dari tahap yang gagal tanpa mengulang transkripsi yang sudah selesai.

Jika download YouTube gagal karena yt-dlp usang:
```powershell
pip install -U yt-dlp          # lokal
# atau, di Docker:
docker compose build --no-cache
```

---

## Konteks terjemahan

File di folder `context/` dibaca ulang **setiap job** — edit saja filenya, tidak perlu restart server.

| File | Isi |
|---|---|
| `context_{grup}_{show}.md` | Style guide + glossary per show |
| `members_{grup}.md` | Roster member per grup |
| `template.ass` | Style ASS (Comic Sans MS 66pt, 1920×1080) |

Saat membuat job, konteks ditentukan secara otomatis dari pilihan grup + show. Tersedia juga kolom **Context Override** (menggantikan seluruh file konteks) dan **Additional Context** (ditambahkan di akhir) untuk penyesuaian per-episode.

---

## Menjalankan test

```powershell
python -m pytest -q
```
