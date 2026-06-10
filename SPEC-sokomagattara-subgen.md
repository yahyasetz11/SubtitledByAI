# SPEC — Sokomagattara SubGen
Platform localhost untuk transkripsi (JP) + translasi (ID) episode *Soko Magattara, Sakurazaka?* dengan output subtitle .ass/.srt bergaya HikaLeon Subs.

---

## 1. Arsitektur & Stack

- **Backend:** Python 3.11+ / FastAPI, serve frontend sendiri.
- **Frontend:** satu file `index.html` + vanilla JS (tanpa build step).
- **Job model:** asinkron via FastAPI `BackgroundTasks`. `POST /jobs` → balas `job_id` segera; progress via SSE `GET /jobs/{id}/events`.
- **Audio tooling:** ffmpeg (ekstraksi, normalisasi, silencedetect, chunking).
- **Downloader:** yt-dlp (pinned di requirements, error message menyarankan `pip install -U yt-dlp` jika gagal).

### Struktur project
```
sokomagattara-subgen/
├── .env
├── app/
│   ├── main.py            # FastAPI + SSE + static
│   ├── pipeline.py        # orkestrasi tahapan job
│   ├── audio.py           # ekstraksi, normalisasi, silencedetect, chunking
│   ├── transcribe.py      # Gemini audio → transkrip JP berstruktur
│   ├── translate.py       # JP → ID (Gemini/GPT/Claude)
│   ├── subtitle.py        # post-processing + render .ass/.srt
│   └── providers.py       # abstraksi klien API + retry
├── context/
│   ├── members.md         # roster member (lihat §4)
│   ├── context.md         # judul, glossary, style guide, few-shot
│   └── template.ass       # header + styles HikaLeon (PlayRes 1920×1080)
├── static/index.html
└── output/{job_id}/
    ├── source.*           # input asli (mp4 jika checkbox simpan dicentang)
    ├── audio.mp3          # audio ternormalisasi
    ├── chunks/            # chunk_001.mp3 + offsets.json
    ├── transcript_jp.json # hasil transkripsi merged
    ├── translated_id.json # hasil translasi merged
    ├── result.ass / result.srt
    └── usage.json         # token usage + estimasi biaya
```

---

## 2. Konfigurasi (.env)

```env
GEMINI_API_KEY=            # WAJIB — transkripsi + translasi default
OPENAI_API_KEY=            # opsional
ANTHROPIC_API_KEY=         # opsional

TRANSCRIBE_MODEL=gemini-2.5-pro
TRANSLATE_MODEL_GEMINI=gemini-2.5-flash     # DEFAULT penerjemah
TRANSLATE_MODEL_OPENAI=gpt-4o
TRANSLATE_MODEL_ANTHROPIC=claude-sonnet-4-6

YTDLP_COOKIES_FILE=        # opsional, untuk video region-locked/membership

# Formatting (lihat §7 — angka dikalibrasi dari .ass HikaLeon aktual)
SUB_MIN_DURATION=0.7
SUB_MAX_DURATION=7.5
SUB_MERGE_GAP=0.5
SUB_CPS_FLAG=25
```

**Aturan startup:** validasi key saat server start. `GEMINI_API_KEY` kosong → exit dengan pesan jelas. Endpoint `GET /config` mengembalikan provider yang tersedia; UI men-disable opsi model tanpa key (tooltip "OPENAI_API_KEY belum diisi di .env").

---

## 3. Input

Dua mode di UI:
1. **Drag-drop file:** mp3, m4a, wav, mp4, mkv. Jika video → `ffmpeg -vn` ekstrak audio.
2. **URL YouTube** + checkbox **"Simpan MP4"**:
   - Unchecked (default): `yt-dlp -f bestaudio` (m4a, lebih cepat).
   - Checked: `bestvideo+bestaudio` → `output/{job_id}/source.mp4` (tombol download di UI); audio diekstrak dari file yang sama — tidak download dua kali.

**Normalisasi:** semua input → mp3 mono 16kHz/64kbps sebelum chunking (cukup untuk ASR, ukuran upload mengecil 3–4×).

---

## 4. File Konteks

### members.md
Roster member (kanji + romaji + nickname) — **sudah diverifikasi per Juni 2026**:
- Morita Hikaru & Tamura Hono: AKTIF (hapus dari tabel graduated).
- Saito Fuyuka: GRADUATED (permanen).
- Perbaiki typo "sesi握手" → "meet & greet"; hapus tab nyasar; **hapus catatan "please verify roster"** (file konteks harus berbicara dengan kepastian, bukan menanam keraguan ke LLM).

### context.md
Tiga bagian: (a) konteks acara (judul, format, MC Sawabe & Tsuchida), (b) glossary istilah, (c) **style guide + few-shot** (8 pasang JP→ID yang sudah disepakati — pronoun aku/kamu, partikel sih/dong/kan/loh/deh, larangan "Anda"/gue-lo/slang Jaksel, keigo ke penonton tetap informal-ramah).

**Injection:** kedua file di-load saat tiap job mulai (hot-reload, edit tanpa restart) dan di-inject ke **kedua tahap** — transkripsi butuh kanji/romaji untuk bias ejaan nama yang benar; translasi butuh nickname + glossary + style.

---

## 5. Chunking

Target chunk: **10–14 menit**, dipotong di natural pause.

1. `ffmpeg silencedetect` threshold awal **-30dB**, cari silence berdurasi **0.5–1.5s** di window menit 10–14 → potong di tengah silence.
2. Kosong → longgarkan threshold bertahap (**-25dB → -20dB**) untuk menangkap momen BGM pelan / pergantian suasana.
3. Tetap kosong → **hard cut di menit 14 + overlap 3–5 detik** ke chunk berikutnya; saat merge, deduplikasi ujaran ganda di zona overlap via kemiripan teks.

Simpan `offsets.json` (offset absolut tiap chunk) untuk koreksi timestamp saat merge.

---

## 6. Tahap API

### 6a. Transkripsi (Gemini 2.5 Pro, via Files API)
- Upload tiap chunk via **Files API** (bukan inline) → URI → request transkripsi. File otomatis terhapus 48 jam.
- Prompt menyertakan members.md + context.md, minta output terstruktur per ujaran:
```json
{"id": 1, "start": "MM:SS.mmm", "end": "MM:SS.mmm",
 "type": "dialogue|narration", "ja": "..."}
```
- `type` = deteksi narator dari audio (VO vs dialog panggung) → dipetakan ke style `Narrator`/`Default` di .ass. Salah-label diterima (koreksi cepat di Aegisub).
- Output terpotong (max tokens) → pecah chunk jadi dua sub-chunk, ulangi.
- Merge: tambahkan offset chunk → timeline absolut.

### 6b. Translasi (default Gemini Flash; opsi GPT/Claude via dropdown)
- Per chunk, kirim **array JSON bernomor** `[{"id":1,"ja":"..."}]`; model wajib balas id sama persis + jumlah sama persis + field `"id_text"`. Pakai JSON mode/structured output bila tersedia.
- **Validasi alignment:** count + id match. Gagal → retry 1× dengan pesan error; masih gagal → fallback terjemahkan per-baris hanya untuk id yang hilang.
- System prompt = context.md (style + few-shot + glossary) + members.md.

---

## 7. Post-Processing Subtitle (deterministik, di backend — bukan LLM)

Angka dikalibrasi dari analisis 388 baris .ass HikaLeon episode #288:

| Aturan | Nilai |
|---|---|
| Durasi minimum | 0.7s — lebih pendek: extend, atau merge dengan tetangga jika gap <0.5s & konteks sama |
| Durasi maksimum | 7.5s — pecah di koma/jeda kalimat |
| Panjang baris | **tanpa wrap otomatis** (gaya HikaLeon: satu baris panjang, hingga ~100 char) |
| CPS | flag ke log "perlu review" jika **>25** — jangan potong otomatis (keputusan editorial milik user) |
| Overlap | geser blok berikutnya mulai setelah blok sebelumnya |
| Merge reaksi | reaksi pendek + lanjutan, gap <0.5s → satu blok ("Aku senang sekali. Impianku jadi kenyataan.") |

---

## 8. Output

- **Default .ass:** render memakai `context/template.ass` (styles Default/Narrator/yt_default, Comic Sans MS, PlayRes 1920×1080) — copy header, isi Events.
- **.srt** tersedia dari data yang sama (toggle/tombol download kedua).
- UI menyediakan download: `result.ass`, `result.srt`, `transcript_jp.json` (untuk QC), dan `source.mp4` bila ada.

---

## 9. Error Handling — Tiga Lapis

1. **Transport:** exponential backoff 3×, hormati `retry-after` (429/5xx/network) — di `providers.py`, berlaku semua provider.
2. **Konten:** validasi skema/alignment → retry 1× dengan feedback error → fallback granular (per-baris untuk translasi; split sub-chunk untuk transkripsi).
3. **Job:** setiap tahap checkpoint ke `output/{job_id}/` → tombol **"Retry from last checkpoint"** di UI; transkripsi (tahap termahal) tidak pernah diulang karena kegagalan tahap setelahnya.

---

## 10. Usage Tracking

- Baca `usage_metadata` dari tiap respons API → akumulasi ke `output/{job_id}/usage.json`.
- UI menampilkan satu baris setelah selesai: `Tokens: 52K in / 12K out (~$0.23)`.
- Ekspektasi biaya: **~$0.20–0.50/episode** (transkripsi Pro ~$0.15–0.30 dominan; translasi Flash <$0.02).

---

## 11. UI (index.html)

Satu halaman:
1. Zona drag-drop + field URL YouTube + checkbox "Simpan MP4".
2. Dropdown penerjemah: **Gemini Flash (default)** / GPT / Claude — opsi tanpa API key disabled + tooltip.
3. Radio output: **.ass (default)** / .srt / keduanya.
4. Timeline progress SSE: Download → Normalize → Chunking → Transcribe (n/N) → Translate (n/N) → Format → Done; tiap tahap ✓/✗.
5. Panel hasil: tombol download + ringkasan usage + daftar baris ter-flag CPS.
6. Tombol "Retry from last checkpoint" muncul saat job gagal.

---

## Pipeline ringkas
```
input (file/URL) → [yt-dlp] → ekstrak audio → normalisasi 16kHz mono
→ silencedetect cascade → chunk 10–14 mnt (+offsets)
→ Gemini Files API → transkrip JP berstruktur (+narrator type) → merge offset
→ translasi batch-JSON per chunk (validasi alignment) → merge
→ post-processing deterministik (§7) → render .ass/.srt → download
```
