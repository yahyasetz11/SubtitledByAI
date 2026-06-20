# Multi-Group Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand SubtitledByAI from Sakurazaka46-only to support Nogizaka46 and Hinatazaka46, with a two-level group→show dropdown, per-group member files, and an "Else/Other" escape hatch for arbitrary content.

**Architecture:** Group selection determines which `members_{group}.md` to load; show selection determines which `context_{preset}.md` to load. Both are resolved in `pipeline._read_context(group, preset)`. The frontend uses two selects (group → show) populated from a hardcoded `GROUPS` dict in `main.py`; show dropdown is hidden when group is "else". "Else" skips both files and uses `additional_context` as the sole LLM context, requiring it to be non-empty.

**Tech Stack:** Python/FastAPI backend, vanilla JS frontend (single HTML file), pytest test suite.

## Global Constraints

- All new context files must be in Indonesian (Bahasa Indonesia), matching the style of `context/context_sokomagattara.md`.
- All new members files must follow the format of `context/members.md` (kanji/romaji/nickname table + graduated section + vocabulary + notes).
- `temperature=0.3`, `max_retries=0` — do not change LLM config.
- Preset key naming: always group-prefixed (e.g., `sakurazaka_sokomagattara`, `nogizaka_kojichuu`).
- No new Python dependencies.
- Test suite must stay green (`pytest` — currently 88 tests).

---

### Task 1: Rename existing context/member files and create all new ones

**Files:**
- Rename: `context/members.md` → `context/members_sakurazaka.md`
- Rename: `context/context_sokomagattara.md` → `context/context_sakurazaka_sokomagattara.md`
- Rename: `context/context_chokosaku.md` → `context/context_sakurazaka_chokosaku.md`
- Keep: `context/context_sakurazaka_channel.md` (already group-prefixed)
- Create: `context/members_nogizaka.md`
- Create: `context/members_hinatazaka.md`
- Create: `context/context_nogizaka_kojichuu.md`
- Create: `context/context_nogizaka_enchouchuu.md`
- Create: `context/context_nogizaka_haishinchuu.md`
- Create: `context/context_hinatazaka_aimashou.md`
- Create: `context/context_hinatazaka_narimashou.md`
- Create: `context/context_hinatazaka_channel.md`

**Interfaces:**
- Produces: 12 files in `context/` with correct names; downstream tasks reference these paths.

- [ ] **Step 1: Rename the three Sakurazaka files**

```powershell
cd C:\Users\YAHYASETZ\Documents\SubtitledByAI
Rename-Item context\members.md context\members_sakurazaka.md
Rename-Item context\context_sokomagattara.md context\context_sakurazaka_sokomagattara.md
Rename-Item context\context_chokosaku.md context\context_sakurazaka_chokosaku.md
```

- [ ] **Step 2: Create `context/members_nogizaka.md`**

Write file with this structure (pre-populated with best-effort data — **user must verify current roster**):

```markdown
# Nogizaka46 Member Reference

This file lists current and graduated members for use during transcription correction and translation.
Always spell names exactly as written here when they appear in subtitles.

---

## Current Members (Active as of June 2026 — please verify)

### 3rd Generation

| Kanji      | Romaji           | Common Nickname |
| ---------- | ---------------- | --------------- |
| 梅澤美波   | Umezawa Minami   | Minami          |
| 向井葉月   | Mukai Hazuki     | Hazuki          |
| 与田祐希   | Yoda Yuki        | Yoda            |
| 中村麗乃   | Nakamura Reno    | Reno            |

### 4th Generation

| Kanji              | Romaji                    | Common Nickname |
| ------------------ | ------------------------- | --------------- |
| 遠藤さくら         | Endo Sakura               | Sakura          |
| 賀喜遥香           | Kaki Haruka               | Kaki            |
| 柴田柚菜           | Shibata Yuna              | Yuna            |
| 早川聖来           | Hayakawa Seiran           | Seiran          |
| 林瑠奈             | Hayashi Luna              | Luna            |
| 北川悠理           | Kitagawa Yuri             | Yuri            |
| 松尾美佑           | Matsuo Miyu               | Miyu            |
| 吉田綾乃クリスティー | Yoshida Ayano Christie   | Christie        |
| 清宮レイ           | Seimiya Rei               | Rei             |
| 佐藤璃果           | Sato Rika                 | Rika            |

### 5th Generation

| Kanji      | Romaji             | Common Nickname |
| ---------- | ------------------ | --------------- |
| 池田瑛紗   | Ikeda Terasa       | Terasa          |
| 井上和     | Inoue Kazu         | Kazu            |
| 岡本姫奈   | Okamoto Hina       | Hina            |
| 小川彩     | Ogawa Aya          | Aya             |
| 川﨑桜     | Kawasaki Sakura    | Sakura          |
| 菅原咲月   | Sugawara Satsuki   | Satsuki         |
| 冨里奈央   | Tomisato Nao       | Nao             |
| 中西アルノ | Nakanishi Aruno    | Aruno           |
| 弓木奈於   | Yumiki Nao         | Yumiki          |
| 五百城茉央 | Ioki Mao           | Mao             |
| 一ノ瀬美空 | Ichinose Misora    | Misora          |
| 矢久保美緒 | Yakubo Mio         | Mio             |

---

## Graduated Members (may appear in archive footage or throwback segments)

| Kanji      | Romaji              | Notes                        |
| ---------- | ------------------- | ---------------------------- |
| 齋藤飛鳥   | Saito Asuka         | Graduated 2023               |
| 山下美月   | Yamashita Mizuki    | Graduated 2023               |
| 秋元真夏   | Akimoto Manatsu     | Graduated 2023               |
| 白石麻衣   | Shiraishi Mai       | Graduated 2020               |
| 西野七瀬   | Nishino Nanase      | Graduated 2019               |
| 松村沙友理 | Matsumura Sayuri    | Graduated 2022               |
| 生田絵梨花 | Ikuta Erika         | Graduated 2022               |
| 堀未央奈   | Hori Miona          | Graduated 2021               |
| 山崎怜奈   | Yamazaki Rena       | Graduated 2024               |

---

## Show-Specific Vocabulary

Common terms in Nogizaka46 content:

| Japanese           | Romaji           | Indonesian Translation                             |
| ------------------ | ---------------- | -------------------------------------------------- |
| 乃木坂工事中       | Nogizaka Kojichuu | Nogizaka Kojichuu (keep title as-is)              |
| 工事中             | Kojichuu         | Kojichuu (keep as-is)                              |
| 乃木坂46          | Nogizaka46       | Nogizaka46 (keep as-is)                            |
| 選抜               | senbatsu         | senbatsu (keep as-is)                              |
| センター           | sentaa           | center / posisi center                             |
| 推し               | oshi             | oshi (keep as-is)                                  |
| キャプテン         | kyaputen         | kapten                                             |
| 先輩               | senpai           | senpai (keep as-is)                                |
| 後輩               | kouhai           | kouhai (keep as-is)                                |
| 収録               | shuuroku         | syuting / rekaman                                  |
| 桜坂               | Sakurazaka       | Sakurazaka (keep as-is)                            |
| 日向坂             | Hinatazaka       | Hinatazaka (keep as-is)                            |
| 運営               | un'ei            | manajemen                                          |
| ファン             | fan              | penggemar / fans                                   |
| 番組               | bangumi          | program / acara                                    |
| 乃木恋             | Nogikoi          | Nogikoi (game title, keep as-is)                   |
| 真夏の全国ツアー   | Manatsu no Zenkoku Tour | tur nasional musim panas                  |

---

## Notes for Translator

- When two members share the same given name, use the surname or nickname to distinguish.
- Nicknames vary by who's speaking — keep the most recognizable form.
- MC Bananaman (バナナマン): 日村勇紀 (Himura Yuuki) called "Himura" and 設楽統 (Shitara Osamu) called "Shitara".
```

- [ ] **Step 3: Create `context/members_hinatazaka.md`**

Write file with this structure (pre-populated — **user must verify current roster**):

```markdown
# Hinatazaka46 Member Reference

This file lists current and graduated members for use during transcription correction and translation.
Always spell names exactly as written here when they appear in subtitles.

---

## Current Members (Active as of June 2026 — please verify)

### 1st Generation

| Kanji      | Romaji           | Common Nickname |
| ---------- | ---------------- | --------------- |
| 金村美玖   | Kanamura Miku    | Miku            |
| 河田陽菜   | Kawata Hina      | Hina            |
| 佐々木久美 | Sasaki Kumi      | Kumi            |

### 2nd Generation

| Kanji        | Romaji              | Common Nickname |
| ------------ | ------------------- | --------------- |
| 石塚瑶季     | Ishizuka Yuki       | Yuki            |
| 上村ひなの   | Kamimura Hinano     | Hinano          |
| 髙橋未来虹   | Takahashi Mikuniko  | Mikuniko        |
| 森本茉莉     | Morimoto Mari       | Mari            |
| 山口陽世     | Yamaguchi Haruyo    | Haruyo          |

### 3rd Generation

| Kanji      | Romaji           | Common Nickname |
| ---------- | ---------------- | --------------- |
| 小西夏菜実 | Konishi Kanami   | Kanami          |
| 平岡海月   | Hiraoka Mizuki   | Mizuki          |
| 山下葉留花 | Yamashita Haruka | Haruka          |

### 4th Generation

| Kanji      | Romaji           | Common Nickname |
| ---------- | ---------------- | --------------- |
| 藤嶌果歩   | Fujiwa Kaho      | Kaho            |
| 正源司陽子 | Shogenji Yoko    | Yoko            |
| 清水理央   | Shimizu Rio      | Rio             |

### 5th Generation

| Kanji      | Romaji           | Common Nickname |
| ---------- | ---------------- | --------------- |
| (未確認 — please fill in 5th gen members) | | |

---

## Graduated Members (may appear in archive footage or throwback segments)

| Kanji      | Romaji           | Notes                     |
| ---------- | ---------------- | ------------------------- |
| 小坂菜緒   | Kosaka Nao       | Graduated 2023            |
| 齊藤京子   | Saito Kyoko      | Graduated 2023            |
| 佐々木美玲 | Sasaki Mirei     | Graduated 2024            |
| 高瀬愛奈   | Takase Mana      | Graduated 2023            |
| 高本彩花   | Takamoto Ayaka   | Graduated 2023            |
| 東村芽依   | Higashimura Mei  | Graduated 2023            |
| 丹生明里   | Nibu Akari       | Graduated 2023            |
| 濱岸ひより | Hamagishi Hiyori | Graduated 2023            |
| 松田好花   | Matsuda Konoka   | Graduated 2024            |
| 宮田愛萌   | Miyata Manamo    | Graduated 2023            |
| 渡邉美穂   | Watanabe Miho    | Graduated 2022            |

---

## Show-Specific Vocabulary

Common terms in Hinatazaka46 content:

| Japanese             | Romaji                    | Indonesian Translation                             |
| -------------------- | ------------------------- | -------------------------------------------------- |
| 日向坂で会いましょう | Hinatazaka de Aimashou    | Hinatazaka de Aimashou (keep title as-is)          |
| ひなあい             | Hina-Ai                   | Hina-Ai (singkatan acara, keep as-is)              |
| 日向坂46            | Hinatazaka46              | Hinatazaka46 (keep as-is)                          |
| 推し                 | oshi                      | oshi (keep as-is)                                  |
| キャプテン           | kyaputen                  | kapten                                             |
| 先輩                 | senpai                    | senpai (keep as-is)                                |
| 後輩                 | kouhai                    | kouhai (keep as-is)                                |
| 収録                 | shuuroku                  | syuting / rekaman                                  |
| 乃木坂               | Nogizaka                  | Nogizaka (keep as-is)                              |
| 桜坂                 | Sakurazaka                | Sakurazaka (keep as-is)                            |
| 運営                 | un'ei                     | manajemen                                          |
| 番組                 | bangumi                   | program / acara                                    |
| おひさま             | Ohisama                   | Ohisama (nama fandom Hinatazaka46, keep as-is)     |

---

## Notes for Translator

- When two members share the same given name, use surname or nickname to distinguish.
- MC Audrey (オードリー): 若林正恭 (Wakabayashi Masayasu) called "Wakabayashi" and 春日俊彰 (Kasuga Toshiaki) called "Kasuga" — for Hinatazaka de Aimashou.
- 5th gen members appear in Hinatazaka de Narimashou — fill in their names once verified.
```

- [ ] **Step 4: Create `context/context_nogizaka_kojichuu.md`**

```markdown
# Translation Context — Nogizaka Kojichuu (乃木坂工事中)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** 乃木坂工事中 (Nogizaka Kojichuu) — dalam subtitle selalu ditulis **"Nogizaka Kojichuu"**, jangan diterjemahkan.
- **Format:** variety show mingguan Nogizaka46 (TV Tokyo, ~24 menit per episode). Berisi segmen permainan, tantangan, talk, dan VTR/narasi.
- **MC:** duo komedian **バナナマン (Bananaman)** — **日村勇紀 (Himura Yuuki)** dipanggil **"Himura"** dan **設楽統 (Shitara Osamu)** dipanggil **"Shitara"**.
- Member Nogizaka46 yang tampil bervariasi tiap episode. Ejaan nama mengikuti `members_nogizaka.md`.
- Acara sering memakai **narasi voice-over (narator)** untuk menjelaskan situasi/aturan segmen.

---

## 2. Glossary Istilah

| Jepang             | Romaji              | Terjemahan Indonesia                                |
|--------------------|---------------------|-----------------------------------------------------|
| 乃木坂工事中       | Nogizaka Kojichuu   | Nogizaka Kojichuu (jangan diterjemahkan)            |
| 選抜               | senbatsu            | senbatsu (biarkan)                                  |
| センター           | sentaa              | center / posisi center                              |
| 推し               | oshi                | oshi (biarkan)                                      |
| 握手会             | akushukai           | meet & greet                                        |
| ライブ             | raibu               | konser / live                                       |
| 振り付け           | furitsuke           | koreografi                                          |
| キャプテン         | kyaputen            | kapten                                              |
| 先輩               | senpai              | senpai (biarkan)                                    |
| 後輩               | kouhai              | kouhai (biarkan)                                    |
| 収録               | shuuroku            | syuting / rekaman                                   |
| 運営               | un'ei               | manajemen                                           |
| ファン             | fan                 | penggemar / fans                                    |
| 番組               | bangumi             | acara / program                                     |
| 罰ゲーム           | batsu game          | hukuman                                             |
| ドッキリ           | dokkiri             | prank                                               |
| テロップ           | teroppu             | teks layar                                          |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya" (kecuali konteks sangat formal), dan JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member dan istilah mengikuti `members_nogizaka.md` dan glossary, ejaan persis.
6. Angka dan satuan ditulis biasa.
7. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: ちょっと待って、それはずるくない？
ID: Eh bentar, itu curang dong?

JP: いやいやいや、絶対違うって！
ID: Nggak nggak nggak, jelas bukan gitu!

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Audio adalah variety show: ada BGM, laugh track, sound effect. Transkripsikan ujaran yang terdengar jelas.
- Nama orang yang terdengar kemungkinan besar adalah member di `members_nogizaka.md` atau MC (Himura/Shitara).
- Tandai `type: "narration"` untuk voice-over narator; selain itu `type: "dialogue"`.
- Teks di layar (telop) tidak perlu ditranskripsikan kecuali dibacakan.
```

- [ ] **Step 5: Create `context/context_nogizaka_enchouchuu.md`**

```markdown
# Translation Context — Nogizaka Kouji Enchouchuu (乃木坂工事中のんびり延長中)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** 乃木坂工事中のんびり延長中 (Nogizaka Kouji Enchouchuu) — dalam subtitle selalu ditulis **"Nogizaka Kouji Enchouchuu"**, jangan diterjemahkan.
- **Format:** extended episode dari Nogizaka Kojichuu — format lebih santai, durasi lebih panjang, sering berisi segmen bonus atau obrolan lebih bebas.
- **MC:** duo komedian **バナナマン (Bananaman)** — **日村勇紀 (Himura Yuuki)** dipanggil **"Himura"** dan **設楽統 (Shitara Osamu)** dipanggil **"Shitara"**.
- Member Nogizaka46 yang tampil mengikuti episode utama. Ejaan nama mengikuti `members_nogizaka.md`.

---

## 2. Glossary Istilah

| Jepang             | Romaji              | Terjemahan Indonesia                                |
|--------------------|---------------------|-----------------------------------------------------|
| 乃木坂工事中       | Nogizaka Kojichuu   | Nogizaka Kojichuu (jangan diterjemahkan)            |
| のんびり延長中     | Enchouchuu          | Enchouchuu (keep as-is)                             |
| 選抜               | senbatsu            | senbatsu (biarkan)                                  |
| センター           | sentaa              | center / posisi center                              |
| 推し               | oshi                | oshi (biarkan)                                      |
| ライブ             | raibu               | konser / live                                       |
| キャプテン         | kyaputen            | kapten                                              |
| 先輩               | senpai              | senpai (biarkan)                                    |
| 後輩               | kouhai              | kouhai (biarkan)                                    |
| 収録               | shuuroku            | syuting / rekaman                                   |
| 運営               | un'ei               | manajemen                                           |
| 番組               | bangumi             | acara / program                                     |
| テロップ           | teroppu             | teks layar                                          |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya" (kecuali konteks formal), JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member dan istilah mengikuti `members_nogizaka.md` dan glossary, ejaan persis.
6. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: ちょっと待って、それはずるくない？
ID: Eh bentar, itu curang dong?

JP: なんか、最近メンバーとご飯行く機会が増えて、嬉しいなって思います。
ID: Akhir-akhir ini makin sering makan bareng member, dan itu bikin aku seneng banget sih.

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Format lebih santai dari episode utama — obrolan lebih bebas, lebih banyak tawa dan interaksi spontan.
- Nama orang kemungkinan besar adalah member di `members_nogizaka.md` atau MC (Himura/Shitara).
- Tandai `type: "narration"` untuk voice-over; selain itu `type: "dialogue"`.
```

- [ ] **Step 6: Create `context/context_nogizaka_haishinchuu.md`**

```markdown
# Translation Context — Nogizaka Haishinchuu (乃木坂配信中)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** 乃木坂配信中 (Nogizaka Haishinchuu) — dalam subtitle selalu ditulis **"Nogizaka Haishinchuu"**, jangan diterjemahkan.
- **Format:** konten YouTube Nogizaka46. Biasanya berupa vlog member, challenge, talk show ringan, atau konten behind-the-scenes tanpa MC tetap.
- **Pembawa acara:** Bervariasi per video — bisa member solo, duo, atau grup kecil. Tidak ada MC tetap.
- Ejaan nama member mengikuti `members_nogizaka.md`.

---

## 2. Glossary Istilah

| Jepang             | Romaji              | Terjemahan Indonesia                                |
|--------------------|---------------------|-----------------------------------------------------|
| 乃木坂配信中       | Nogizaka Haishinchuu | Nogizaka Haishinchuu (jangan diterjemahkan)        |
| 配信               | haishin             | streaming / siaran                                  |
| 推し               | oshi                | oshi (biarkan)                                      |
| ライブ             | raibu               | konser / live                                       |
| 先輩               | senpai              | senpai (biarkan)                                    |
| 後輩               | kouhai              | kouhai (biarkan)                                    |
| 収録               | shuuroku            | syuting / rekaman                                   |
| 運営               | un'ei               | manajemen                                           |
| チャンネル         | channeru            | channel                                             |
| 登録               | touroku             | subscribe / berlangganan                            |
| コメント           | komento             | komentar                                            |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya" (kecuali konteks sangat formal), JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member mengikuti `members_nogizaka.md`, ejaan persis.
6. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: 皆さん、こんにちは！今日は一人でお届けします！
ID: Halo semua! Hari ini aku bakal nemenin kalian sendirian nih!

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: チャンネル登録よろしくお願いします！
ID: Jangan lupa subscribe channel-nya ya!

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Konten YouTube: audio sering lebih spontan, mungkin ada musik latar/editing. Transkripsikan ujaran yang terdengar jelas.
- Tidak ada MC tetap — perhatikan siapa yang berbicara berdasarkan konteks atau Additional Context.
- Tandai `type: "narration"` untuk voice-over/narasi editor; selain itu `type: "dialogue"`.
```

- [ ] **Step 7: Create `context/context_hinatazaka_aimashou.md`**

```markdown
# Translation Context — Hinatazaka de Aimashou (日向坂で会いましょう)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** 日向坂で会いましょう (Hinatazaka de Aimashou) — disingkat **"ひなあい" (Hina-Ai)**. Dalam subtitle selalu ditulis **"Hinatazaka de Aimashou"**, jangan diterjemahkan.
- **Format:** variety show mingguan Hinatazaka46 (TV Tokyo, ~24 menit per episode). Berisi segmen permainan, tantangan, talk, dan VTR.
- **MC:** duo komedian **オードリー (Audrey)** — **若林正恭 (Wakabayashi Masayasu)** dipanggil **"Wakabayashi"** dan **春日俊彰 (Kasuga Toshiaki)** dipanggil **"Kasuga"**.
- Member Hinatazaka46 yang tampil bervariasi tiap episode. Ejaan nama mengikuti `members_hinatazaka.md`.
- Acara sering memakai **narasi voice-over (narator)** untuk menjelaskan situasi/aturan segmen.

---

## 2. Glossary Istilah

| Jepang               | Romaji                 | Terjemahan Indonesia                               |
|----------------------|------------------------|----------------------------------------------------|
| 日向坂で会いましょう | Hinatazaka de Aimashou | Hinatazaka de Aimashou (jangan diterjemahkan)      |
| ひなあい             | Hina-Ai                | Hina-Ai (singkatan, keep as-is)                    |
| おひさま             | Ohisama                | Ohisama (nama fandom, keep as-is)                  |
| 選抜                 | senbatsu               | senbatsu (biarkan)                                 |
| センター             | sentaa                 | center / posisi center                             |
| 推し                 | oshi                   | oshi (biarkan)                                     |
| 握手会               | akushukai              | meet & greet                                       |
| ライブ               | raibu                  | konser / live                                      |
| 振り付け             | furitsuke              | koreografi                                         |
| キャプテン           | kyaputen               | kapten                                             |
| 先輩                 | senpai                 | senpai (biarkan)                                   |
| 後輩                 | kouhai                 | kouhai (biarkan)                                   |
| 収録                 | shuuroku               | syuting / rekaman                                  |
| 運営                 | un'ei                  | manajemen                                          |
| 番組                 | bangumi                | acara / program                                    |
| 罰ゲーム             | batsu game             | hukuman                                            |
| ドッキリ             | dokkiri                | prank                                              |
| テロップ             | teroppu                | teks layar                                         |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya" (kecuali konteks sangat formal), JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member dan istilah mengikuti `members_hinatazaka.md` dan glossary, ejaan persis.
6. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: ちょっと待って、それはずるくない？
ID: Eh bentar, itu curang dong?

JP: いやいやいや、絶対違うって！
ID: Nggak nggak nggak, jelas bukan gitu!

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Audio adalah variety show: ada BGM, laugh track, sound effect. Transkripsikan ujaran yang terdengar jelas.
- Nama orang kemungkinan besar adalah member di `members_hinatazaka.md` atau MC (Wakabayashi/Kasuga).
- Tandai `type: "narration"` untuk voice-over narator; selain itu `type: "dialogue"`.
- Teks di layar (telop) tidak perlu ditranskripsikan kecuali dibacakan.
```

- [ ] **Step 8: Create `context/context_hinatazaka_narimashou.md`**

```markdown
# Translation Context — Hinatazaka de Narimashou (日向坂でなりましょう)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** 日向坂でなりましょう (Hinatazaka de Narimashou) — dalam subtitle selalu ditulis **"Hinatazaka de Narimashou"**, jangan diterjemahkan.
- **Format:** variety show untuk member **generasi ke-5 (5期生)** Hinatazaka46. Format lebih ringan dan santai, berfokus pada perkenalan dan pertumbuhan member baru.
- **MC:** [Konfirmasi diperlukan — isi setelah verifikasi]. Ejaan nama MC mengikuti format Kanji (Romaji).
- Member yang tampil adalah member generasi ke-5. Ejaan nama mengikuti `members_hinatazaka.md`.

---

## 2. Glossary Istilah

| Jepang               | Romaji                  | Terjemahan Indonesia                               |
|----------------------|-------------------------|----------------------------------------------------|
| 日向坂でなりましょう | Hinatazaka de Narimashou | Hinatazaka de Narimashou (jangan diterjemahkan)   |
| 5期生                | goukisei                | member generasi ke-5                               |
| おひさま             | Ohisama                 | Ohisama (nama fandom, keep as-is)                  |
| 推し                 | oshi                    | oshi (biarkan)                                     |
| センター             | sentaa                  | center / posisi center                             |
| 先輩                 | senpai                  | senpai (biarkan)                                   |
| 後輩                 | kouhai                  | kouhai (biarkan)                                   |
| 収録                 | shuuroku                | syuting / rekaman                                  |
| 運営                 | un'ei                   | manajemen                                          |
| 番組                 | bangumi                 | acara / program                                    |
| テロップ             | teroppu                 | teks layar                                         |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya", JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member mengikuti `members_hinatazaka.md`, ejaan persis.
6. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: なんか、最近メンバーとご飯行く機会が増えて、嬉しいなって思います。
ID: Akhir-akhir ini makin sering makan bareng member, dan itu bikin aku seneng banget sih.

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Konten berfokus pada member 5th gen yang masih baru — mereka mungkin berbicara lebih formal atau gugup.
- Nama orang kemungkinan besar adalah member generasi ke-5 di `members_hinatazaka.md`.
- Tandai `type: "narration"` untuk voice-over; selain itu `type: "dialogue"`.
```

- [ ] **Step 9: Create `context/context_hinatazaka_channel.md`**

```markdown
# Translation Context — Hinatazaka Channel (日向坂チャンネル)

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** Hinatazaka Channel (日向坂チャンネル) — dalam subtitle selalu ditulis **"Hinatazaka Channel"**, jangan diterjemahkan.
- **Format:** konten YouTube Hinatazaka46. Biasanya berupa vlog member, challenge, talk santai, atau behind-the-scenes tanpa MC tetap.
- **Pembawa acara:** Bervariasi per video — bisa member solo, duo, atau grup kecil. Tidak ada MC tetap.
- Ejaan nama member mengikuti `members_hinatazaka.md`.

---

## 2. Glossary Istilah

| Jepang               | Romaji           | Terjemahan Indonesia                               |
|----------------------|------------------|----------------------------------------------------|
| 日向坂チャンネル     | Hinatazaka Channel | Hinatazaka Channel (jangan diterjemahkan)        |
| おひさま             | Ohisama          | Ohisama (nama fandom, keep as-is)                  |
| 配信                 | haishin          | streaming / siaran                                 |
| 推し                 | oshi             | oshi (biarkan)                                     |
| 先輩                 | senpai           | senpai (biarkan)                                   |
| 後輩                 | kouhai           | kouhai (biarkan)                                   |
| 収録                 | shuuroku         | syuting / rekaman                                  |
| チャンネル           | channeru         | channel                                            |
| 登録                 | touroku          | subscribe / berlangganan                           |
| コメント             | komento          | komentar                                           |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya", JANGAN pakai "gue/lo".
2. Partikel percakapan boleh: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu.
4. Reaksi spontan: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih".
5. Nama member mengikuti `members_hinatazaka.md`, ejaan persis.
6. Jangan menambah atau membuang informasi.

### Contoh gaya (few-shot)

JP: 皆さん、こんにちは！今日は一人でお届けします！
ID: Halo semua! Hari ini aku bakal nemenin kalian sendirian nih!

JP: チャンネル登録よろしくお願いします！
ID: Jangan lupa subscribe channel-nya ya!

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

---

## 4. Catatan untuk Tahap Transkripsi

- Konten YouTube: audio lebih spontan. Transkripsikan ujaran yang terdengar jelas.
- Tidak ada MC tetap — gunakan Additional Context untuk mengetahui siapa yang tampil.
- Tandai `type: "narration"` untuk narasi editor; selain itu `type: "dialogue"`.
```

- [ ] **Step 10: Verify all 12 files exist**

```powershell
Get-ChildItem context\ -Filter "context_*.md" | Select-Object Name
Get-ChildItem context\ -Filter "members_*.md" | Select-Object Name
```

Expected output — context files:
```
context_nogizaka_kojichuu.md
context_nogizaka_enchouchuu.md
context_nogizaka_haishinchuu.md
context_hinatazaka_aimashou.md
context_hinatazaka_narimashou.md
context_hinatazaka_channel.md
context_sakurazaka_sokomagattara.md
context_sakurazaka_chokosaku.md
context_sakurazaka_channel.md
```

Expected members files:
```
members_sakurazaka.md
members_nogizaka.md
members_hinatazaka.md
```

- [ ] **Step 11: Commit**

```
git add context/
git commit -m "feat: add Nogizaka46 and Hinatazaka46 context/member files, rename Sakurazaka files to group-prefixed"
```

---

### Task 2: Update `pipeline._read_context()` + pipeline tests

**Files:**
- Modify: `app/pipeline.py:116-123` (`_read_context` function + 2 call sites at lines ~198 and ~269)
- Modify: `tests/test_pipeline.py` (update existing context tests, add new ones)

**Interfaces:**
- Consumes: `context/members_{group}.md`, `context/context_{preset}.md` (Task 1 files)
- Produces: `_read_context(group: str | None, preset: str | None) -> tuple[str, str]`
  - Returns `("", "")` when `group == "else"` or `group` is falsy
  - Returns `(context_md, members_md)` for valid group+preset
  - Raises `FileNotFoundError` if `context_{preset}.md` doesn't exist

- [ ] **Step 1: Write the failing tests first**

In `tests/test_pipeline.py`, add these tests after the existing `test_context_preset_param_loads_correct_file` test:

```python
def test_read_context_loads_group_prefixed_members(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_sokomagattara.md").write_text("SOKO-CTX", encoding="utf-8")
    (ctx_dir / "members_sakurazaka.md").write_text("SAKURA-MEMBERS", encoding="utf-8")
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        received["members_md"] = members
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="sakurazaka_sokomagattara"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == "SOKO-CTX"
    assert received.get("members_md") == "SAKURA-MEMBERS"


def test_read_context_else_returns_empty_strings(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        received["members_md"] = members
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="else", context_preset="",
        additional_context="Custom context for a random vlog"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == ""
    assert received.get("members_md") == ""
```

Also update `test_context_preset_param_loads_correct_file` to use new group-prefixed names:

```python
def test_context_preset_param_loads_correct_file(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_chokosaku.md").write_text("CHOKOSAKU-CTX", encoding="utf-8")
    (ctx_dir / "members_sakurazaka.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    received = {}
    orig = transcribe.build_transcribe_prompts
    def spy(ctx, members, additional_context=None):
        received["context_md"] = ctx
        return orig(ctx, members, additional_context=additional_context)
    monkeypatch.setattr("app.transcribe.build_transcribe_prompts", spy)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="sakurazaka_chokosaku"
    ))
    pipeline.run_job(job.id)
    assert received.get("context_md") == "CHOKOSAKU-CTX"
```

Also update `test_missing_context_preset_fails_job`:

```python
def test_missing_context_preset_fails_job(env, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    monkeypatch.setattr(pipeline, "CONTEXT_DIR", ctx_dir)

    job = pipeline.create_job(make_params(
        group="sakurazaka", context_preset="nonexistent"
    ))
    pipeline.run_job(job.id)
    assert job.status == "failed"
    assert "nonexistent" in (job.error or "")
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_pipeline.py -k "context" -v
```

Expected: FAIL — `_read_context` still has old signature.

- [ ] **Step 3: Update `_read_context` in `app/pipeline.py`**

Replace lines 116–123:

```python
def _read_context(group: str | None, preset: str | None) -> tuple[str, str]:
    if not group or group == "else":
        return "", ""
    members_path = CONTEXT_DIR / f"members_{group}.md"
    members_md = members_path.read_text(encoding="utf-8") if members_path.exists() else ""
    if not preset:
        return "", members_md
    path = CONTEXT_DIR / f"context_{preset}.md"
    if not path.exists():
        raise FileNotFoundError(f"Context preset '{preset}' tidak ditemukan: {path}")
    context_md = path.read_text(encoding="utf-8")
    return context_md, members_md
```

- [ ] **Step 4: Update both call sites in `app/pipeline.py`**

In `_stage_transcribe` (~line 198), replace:
```python
context_md, members_md = _read_context(job.params.get("context_preset"))
```
with:
```python
context_md, members_md = _read_context(
    job.params.get("group"), job.params.get("context_preset")
)
```

In `_stage_translate` (~line 269), replace:
```python
context_md, members_md = _read_context(job.params.get("context_preset"))
```
with:
```python
context_md, members_md = _read_context(
    job.params.get("group"), job.params.get("context_preset")
)
```

- [ ] **Step 5: Run the context tests to confirm they pass**

```
pytest tests/test_pipeline.py -k "context" -v
```

Expected: all PASS.

- [ ] **Step 6: Run the full test suite**

```
pytest -v
```

Expected: all green. If any test references `members.md` directly, update it to `members_sakurazaka.md`.

- [ ] **Step 7: Commit**

```
git add app/pipeline.py tests/test_pipeline.py
git commit -m "feat: update _read_context to accept group+preset, handle else case"
```

---

### Task 3: Update `main.py` API + API tests

**Files:**
- Modify: `app/main.py` (add `GROUPS` dict, update title, restructure `/api/contexts`, add `group` form param)
- Modify: `tests/test_api.py` (update stale `/api/contexts` tests, add group param tests)

**Interfaces:**
- Produces: `GET /api/contexts` → `{"groups": [{"id", "label", "shows": [{"id", "label"}]}]}`
- Produces: `POST /api/jobs` now accepts `group: str = Form("sakurazaka")`
- `params` dict now includes `"group": str`

- [ ] **Step 1: Write the failing tests first**

In `tests/test_api.py`, replace the four stale `/api/contexts` tests (lines ~141–168) and add a `group` param test:

```python
def test_get_contexts_returns_groups_structure(client):
    resp = client.get("/api/contexts")
    assert resp.status_code == 200
    body = resp.json()
    assert "groups" in body
    group_ids = {g["id"] for g in body["groups"]}
    assert group_ids == {"sakurazaka", "nogizaka", "hinatazaka", "else"}


def test_get_contexts_each_group_has_shows(client):
    body = client.get("/api/contexts").json()
    groups = {g["id"]: g for g in body["groups"]}
    assert len(groups["sakurazaka"]["shows"]) == 3
    assert len(groups["nogizaka"]["shows"]) == 3
    assert len(groups["hinatazaka"]["shows"]) == 3
    assert groups["else"]["shows"] == []


def test_get_contexts_show_ids_are_group_prefixed(client):
    body = client.get("/api/contexts").json()
    groups = {g["id"]: g for g in body["groups"]}
    saka_ids = {s["id"] for s in groups["sakurazaka"]["shows"]}
    assert "sakurazaka_sokomagattara" in saka_ids
    assert "sakurazaka_chokosaku" in saka_ids
    assert "sakurazaka_channel" in saka_ids
    nogi_ids = {s["id"] for s in groups["nogizaka"]["shows"]}
    assert "nogizaka_kojichuu" in nogi_ids
    hina_ids = {s["id"] for s in groups["hinatazaka"]["shows"]}
    assert "hinatazaka_aimashou" in hina_ids


def test_create_job_stores_group_param(client, fake_run):
    resp = post_url_job(client, group="nogizaka",
                        context_preset="nogizaka_kojichuu")
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params.get("group") == "nogizaka"
    assert job.params.get("context_preset") == "nogizaka_kojichuu"


def test_create_job_group_defaults_to_sakurazaka(client, fake_run):
    resp = post_url_job(client)
    assert resp.status_code == 200
    job = pipeline.JOBS[resp.json()["job_id"]]
    assert job.params.get("group") == "sakurazaka"
```

Also update `test_get_context_with_preset_param` — the preset name changes from `sokomagattara` to `sakurazaka_sokomagattara`:

```python
def test_get_context_with_preset_param(client, monkeypatch, tmp_path):
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir()
    (ctx_dir / "context_sakurazaka_sokomagattara.md").write_text("# Soko Context", encoding="utf-8")
    monkeypatch.setattr(main, "CONTEXT_DIR", ctx_dir)
    resp = client.get("/api/context?preset=sakurazaka_sokomagattara")
    assert resp.status_code == 200
    assert resp.json()["context_md"] == "# Soko Context"
```

Also update `test_get_context_unknown_preset_returns_404` — no file-system change needed there, just ensure it still passes.

Also update the `_write_job` helper to include `group`:

```python
def _write_job(tmp_path, job_id, status, original_filename=None, url=None):
    job_dir = tmp_path / job_id
    (job_dir / "chunks").mkdir(parents=True, exist_ok=True)
    params = {"source": "file" if original_filename else "url",
              "url": url, "translator": "gemini", "output_format": "ass",
              "save_mp4": False, "original_filename": original_filename,
              "group": "sakurazaka", "context_preset": "sakurazaka_sokomagattara",
              "context_override": None, "additional_context": None}
    meta = {"id": job_id, "params": params, "status": status,
            "error": None, "events": []}
    (job_dir / "job.json").write_text(
        _json.dumps(meta, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_api.py -k "contexts or group or preset" -v
```

Expected: FAIL — old `contexts` endpoint returns `{"contexts": [...]}` not `{"groups": [...]}`.

- [ ] **Step 3: Update `app/main.py`**

After the imports, add the `GROUPS` dict and update the FastAPI title:

```python
app = FastAPI(title="SubtitledByAI")

GROUPS: dict[str, dict] = {
    "sakurazaka": {
        "label": "Sakurazaka46",
        "shows": [
            {"id": "sakurazaka_sokomagattara", "label": "Sokomagattara Sakurazaka"},
            {"id": "sakurazaka_chokosaku", "label": "Choko-Saku"},
            {"id": "sakurazaka_channel", "label": "Sakurazaka Channel"},
        ],
    },
    "nogizaka": {
        "label": "Nogizaka46",
        "shows": [
            {"id": "nogizaka_kojichuu", "label": "Nogizaka Kojichuu"},
            {"id": "nogizaka_enchouchuu", "label": "Nogizaka Kouji Enchouchuu"},
            {"id": "nogizaka_haishinchuu", "label": "Nogizaka Haishinchuu"},
        ],
    },
    "hinatazaka": {
        "label": "Hinatazaka46",
        "shows": [
            {"id": "hinatazaka_aimashou", "label": "Hinatazaka de Aimashou"},
            {"id": "hinatazaka_narimashou", "label": "Hinatazaka de Narimashou"},
            {"id": "hinatazaka_channel", "label": "Hinatazaka Channel"},
        ],
    },
}
```

Replace the `list_contexts` function:

```python
@app.get("/api/contexts")
def list_contexts() -> dict:
    groups = [
        {"id": gid, "label": gdata["label"], "shows": gdata["shows"]}
        for gid, gdata in GROUPS.items()
    ]
    groups.append({"id": "else", "label": "Else / Other", "shows": []})
    return {"groups": groups}
```

In the `create_job` endpoint signature, add `group` after `save_mp4`:

```python
group: str = Form("sakurazaka"),
context_preset: str = Form("sakurazaka_sokomagattara"),
```

In the `params` dict inside `create_job`, add `group`:

```python
params = {"source": source, "url": url.strip() or None,
          "translator": translator, "output_format": output_format,
          "save_mp4": save_mp4, "original_filename": upload_name,
          "group": group.strip() or "sakurazaka",
          "context_preset": context_preset.strip() or "sakurazaka_sokomagattara",
          "context_override": context_override.strip() or None,
          "additional_context": additional_context.strip() or None}
```

- [ ] **Step 4: Run the API tests**

```
pytest tests/test_api.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the full suite**

```
pytest -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```
git add app/main.py tests/test_api.py
git commit -m "feat: add GROUPS dict, restructure /api/contexts to return groups, add group form param"
```

---

### Task 4: Update `static/index.html` frontend

**Files:**
- Modify: `static/index.html` (title, h1, tagline, two-level dropdown, JS logic, form submission)

**Interfaces:**
- Consumes: `GET /api/contexts` → `{"groups": [...]}` (Task 3)
- Consumes: `GET /api/context?preset=...` (unchanged)
- Produces: `POST /api/jobs` with `group` field added to FormData

- [ ] **Step 1: Update title, h1, and tagline**

In `static/index.html`, make these three replacements:

1. `<title>Sokomagattara SubGen</title>` → `<title>SubtitledByAI</title>`

2. `<h1>Sokomagattara <span class="gradient-text">SubGen</span></h1>` → `<h1>Subtitled<span class="gradient-text">ByAI</span></h1>`

3. `<p class="tagline">Transkripsi JP → Terjemahan ID → Subtitle .ass / .srt</p>` → `<p class="tagline">Subtitle otomatis untuk konten Sakurazaka46, Nogizaka46, dan Hinatazaka46</p>`

- [ ] **Step 2: Replace the single context_preset field with two-level group→show dropdowns**

Replace the entire `<div class="field">` block containing `context_preset` (lines ~556–570):

```html
<div class="field">
  <label class="field-label" for="group_select">Grup</label>
  <select id="group_select" style="margin-bottom:8px;">
    <option value="">Memuat...</option>
  </select>
</div>

<div class="field" id="show-field">
  <label class="field-label" for="context_preset">Acara / Konten</label>
  <select id="context_preset" style="margin-bottom:8px;">
    <option value="">Pilih grup terlebih dahulu</option>
  </select>
  <button type="button" class="collapsible-toggle" id="ctx-toggle">
    <span class="ctx-arrow" id="ctx-arrow">▶</span>
    <span>Lihat / Edit konteks</span>
  </button>
  <div class="collapsible-body" id="ctx-body">
    <textarea id="context_override" rows="8"
      style="margin-top:8px; font-family:'JetBrains Mono',Consolas,monospace; font-size:.78rem;"
      placeholder="Memuat konteks..."></textarea>
  </div>
</div>
```

- [ ] **Step 3: Update the `additional_context` field label and placeholder to hint at "Else" requirement**

Replace the `additional_context` field (lines ~550–554):

```html
<div class="field">
  <label class="field-label" for="additional_context">
    Konteks Video
    <span id="ctx-required-badge" class="hidden"
      style="color:#ff6b6b;font-size:.8rem;margin-left:6px;">* Wajib diisi</span>
    <span id="ctx-optional-badge" style="color:var(--muted2);font-size:.8rem;margin-left:6px;">(Opsional)</span>
  </label>
  <textarea id="additional_context" rows="2"
    placeholder="Contoh: Vlog Moriya Rena memancing di Kanagawa. Tulis konteks singkat agar Gemini tidak salah tebak."></textarea>
  <p id="ctx-else-hint" class="hidden"
    style="color:#ff6b6b;font-size:.82rem;margin-top:4px;">
    Pilihan "Else / Other" tidak menggunakan preset — wajib isi konteks di atas.
  </p>
</div>
```

- [ ] **Step 4: Replace the JavaScript context-loading section**

Replace the entire `// ── Context preset selector` block (~lines 945–976) with:

```javascript
// ── Context group + show selector ────────────────────────────────
let groupsData = [];  // [{id, label, shows:[{id,label}]}]

async function loadContextForPreset(preset) {
  if (!preset) { $("context_override").value = ""; return; }
  try {
    const data = await (await fetch(`/api/context?preset=${encodeURIComponent(preset)}`)).json();
    $("context_override").value = data.context_md || "";
  } catch (e) {
    $("context_override").placeholder = "Gagal memuat konteks: " + e;
  }
}

function onGroupChange() {
  const groupId = $("group_select").value;
  const group = groupsData.find(g => g.id === groupId);
  const isElse = groupId === "else";

  // Show/hide show dropdown
  $("show-field").classList.toggle("hidden", isElse);

  // Show/hide required badge and hint
  $("ctx-required-badge").classList.toggle("hidden", !isElse);
  $("ctx-optional-badge").classList.toggle("hidden", isElse);
  $("ctx-else-hint").classList.toggle("hidden", !isElse);

  if (isElse) {
    $("context_preset").value = "";
    $("context_override").value = "";
    return;
  }

  // Populate show dropdown
  const sel = $("context_preset");
  sel.innerHTML = "";
  if (group && group.shows.length > 0) {
    for (const show of group.shows) {
      const opt = document.createElement("option");
      opt.value = show.id;
      opt.textContent = show.label;
      sel.appendChild(opt);
    }
    loadContextForPreset(sel.value);
  }
}

async function loadGroups() {
  try {
    const data = await (await fetch("/api/contexts")).json();
    groupsData = data.groups || [];
    const sel = $("group_select");
    sel.innerHTML = "";
    for (const g of groupsData) {
      const opt = document.createElement("option");
      opt.value = g.id;
      opt.textContent = g.label;
      if (g.id === "sakurazaka") opt.selected = true;
      sel.appendChild(opt);
    }
    onGroupChange();
  } catch (e) {
    $("group_select").innerHTML = '<option value="">Gagal memuat grup</option>';
  }
}

$("group_select").addEventListener("change", onGroupChange);
$("context_preset").addEventListener("change", () => {
  loadContextForPreset($("context_preset").value);
});
```

- [ ] **Step 5: Update form submission to include `group` and validate "Else"**

Replace the submit handler block (lines ~836–866) — specifically the lines that build `fd`:

Add `group` to FormData and add the "Else" validation. Replace the block between `e.preventDefault();` and `const btn = $("start-btn");`:

```javascript
  e.preventDefault();
  const source = document.querySelector('input[name="source"]:checked').value;
  const groupVal = $("group_select").value;
  const additionalCtx = $("additional_context").value.trim();

  // Validate: Else requires additional context
  if (groupVal === "else" && !additionalCtx) {
    $("additional_context").focus();
    $("additional_context").style.outline = "2px solid #ff6b6b";
    setTimeout(() => { $("additional_context").style.outline = ""; }, 2000);
    return;
  }

  const fd = new FormData();
  fd.append("source", source);
  fd.append("translator", $("translator").value);
  fd.append("output_format", $("output_format").value);
  fd.append("additional_context", additionalCtx);
  fd.append("group", groupVal);
  fd.append("context_preset", $("context_preset").value);
  fd.append("context_override", $("context_override").value.trim());
```

- [ ] **Step 6: Update `loadContextPresets()` call at page-init to `loadGroups()`**

Find the line that calls `loadContextPresets()` at page init (near bottom of the script) and replace it with `loadGroups()`.

- [ ] **Step 7: Manually test in browser**

Start the server: `uvicorn app.main:app --reload`

Check:
1. Title shows "SubtitledByAI" in browser tab
2. Header shows "SubtitledByAI" with gradient
3. Group dropdown shows Sakurazaka46 / Nogizaka46 / Hinatazaka46 / Else / Other
4. Selecting Sakurazaka46 shows 3 shows; selecting Nogizaka46 shows 3 shows
5. Selecting "Else / Other" hides the show dropdown, shows red "* Wajib diisi" badge
6. Submitting with "Else" and empty additional context → field flashes red, does not submit
7. Changing group loads the first show's context in the edit textarea

- [ ] **Step 8: Commit**

```
git add static/index.html
git commit -m "feat: two-level group→show dropdown, Else validation, rebrand to SubtitledByAI"
```
