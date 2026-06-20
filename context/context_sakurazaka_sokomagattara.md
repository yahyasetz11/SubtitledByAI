# Translation Context — Soko Magattara, Sakurazaka?

File ini di-inject ke system prompt pada tahap transkripsi DAN translasi.
Semua informasi di sini bersifat pasti. Ikuti tanpa pengecualian.

---

## 1. Tentang Acara

- **Judul:** そこ曲がったら、櫻坂？ (Soko Magattara, Sakurazaka?) — dalam subtitle selalu ditulis **"Sokomagattara Sakurazaka"**, jangan diterjemahkan.
- **Format:** variety show mingguan Sakurazaka46 (TV Tokyo, ~24 menit per episode). Berisi segmen permainan, tantangan, talk, lokasi luar studio, dan VTR/narasi.
- **MC:** duo komedian **Sawabe Yu** (澤部佑, dari Haraichi) dan **Tsuchida Teruyuki** (土田晃之). Dalam subtitle cukup tulis **"Sawabe"** dan **"Tsuchida"**.
- Member Sakurazaka46 yang tampil bervariasi tiap episode. Ejaan nama mengikuti `members.md`.
- Acara sering memakai **narasi voice-over (narator)** untuk menjelaskan situasi/aturan segmen — bedakan dari dialog panggung (field `type`: `narration` vs `dialogue`).

---

## 2. Glossary Istilah

Gunakan terjemahan berikut secara konsisten:

| Jepang | Romaji | Terjemahan Indonesia |
|---|---|---|
| そこ曲がったら、櫻坂？ | Sokomagattara Sakurazaka? | Sokomagattara Sakurazaka (jangan diterjemahkan) |
| 合宿 | gasshuku | training camp |
| 選抜 | senbatsu | senbatsu (biarkan, istilah dikenal fans) |
| センター | sentaa | center / posisi center |
| 推し | oshi | oshi (biarkan) |
| 握手会 | akushukai | meet & greet |
| ライブ | raibu | konser / live |
| 振り付け | furitsuke | koreografi |
| 歌詞 | kashi | lirik |
| キャプテン | kyaputen | kapten |
| 先輩 | senpai | senpai (biarkan) |
| 後輩 | kouhai | kouhai (biarkan) |
| 収録 | shuuroku | syuting / rekaman |
| 乃木坂 | Nogizaka | Nogizaka (biarkan) |
| 欅坂 | Keyakizaka | Keyakizaka (biarkan) |
| 櫻坂 | Sakurazaka | Sakurazaka (biarkan) |
| 運営 | un'ei | manajemen |
| ファン | fan | penggemar / fans |
| Buddies | — | Buddies (nama fandom Sakurazaka46, biarkan) |
| 番組 | bangumi | acara / program |
| 罰ゲーム | batsu game | hukuman |
| ドッキリ | dokkiri | prank |
| VTR | — | video / VTR |
| テロップ | teroppu | teks layar |

---

## 3. Style Guide Terjemahan

Target: **Bahasa Indonesia informal yang natural — santai tapi tidak terlalu slang.**

### Aturan wajib
1. Pronoun: **"aku/kamu"**. JANGAN PERNAH pakai "Anda", "saya" (kecuali konteks sangat formal seperti pengumuman resmi), dan JANGAN pakai "gue/lo".
2. Partikel percakapan boleh dan dianjurkan secukupnya: **sih, dong, kan, loh, deh, nih, ya**.
3. DILARANG slang Jaksel / campuran Inggris yang tidak perlu ("literally", "which is", "you know").
4. Kalimat keigo/sopan ke penonton tetap diterjemahkan **informal-ramah**, bukan formal kaku.
5. Reaksi spontan diterjemahkan hidup: えっ→"Eh!?", マジで→"Serius?!", うそ→"Bohong!/Masa sih?!", やばい→"Gawat/Gila sih" (sesuai konteks).
6. Nama member dan istilah mengikuti `members.md` dan glossary di atas, ejaan persis.
7. Angka dan satuan ditulis biasa (10 Juni, 30 besar, 15 menit).
8. Jangan menambah atau membuang informasi; satu ujaran sumber = satu entri terjemahan (jaga alignment id).

### Contoh gaya (few-shot) — tirukan nada ini

JP: えー、マジで！？それ初めて聞いたんだけど！
ID: Hah, serius!? Aku baru denger itu loh!

JP: ちょっと待って、それはずるくない？
ID: Eh bentar, itu curang dong?

JP: いやいやいや、絶対違うって！
ID: Nggak nggak nggak, jelas bukan gitu!

JP: 続きましては、こちらのコーナーです！
ID: Selanjutnya, kita masuk ke segmen ini!

JP: なんか、最近メンバーとご飯行く機会が増えて、嬉しいなって思います。
ID: Akhir-akhir ini makin sering makan bareng member, dan itu bikin aku seneng banget sih.

JP: それな！うちもずっとそう思ってた！
ID: Nah itu! Aku juga dari dulu mikir gitu!

JP: お疲れ様でした〜！
ID: Kerja bagus hari ini~!

JP: 皆さん、ぜひチェックしてみてください！
ID: Teman-teman, jangan lupa dicek ya!

---

## 4. Catatan untuk Tahap Transkripsi

- Audio adalah variety show: ada BGM, laugh track, sound effect, dan tumpang-tindih suara. Transkripsikan ujaran yang terdengar jelas; abaikan backchannel yang tidak bermakna (うん, はい pendek) kecuali menjadi jawaban penting.
- Nama orang yang terdengar kemungkinan besar adalah member di `members.md` atau MC (Sawabe/Tsuchida) — utamakan ejaan dari daftar tersebut.
- Tandai `type: "narration"` untuk voice-over narator (nada baca, tanpa suasana studio); selain itu `type: "dialogue"`.
- Teks di layar (telop) tidak perlu ditranskripsikan kecuali dibacakan.