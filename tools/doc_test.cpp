// tools/doc_writer.h on a sample that uses every feature: writes <out>.docx and <out>.pdf (and <out>-letter.pdf in
// landscape Letter with a serif font), for opening in Word and checking with a PDF reader.
//   doc_test <out path without extension>
#include "doc_writer.h"
#include <fstream>
#include <iostream>

static const char* kSample = R"MD(# Notulen Rapat Tim Produk

**Tanggal:** Senin, 26 Oktober 2026 · **Waktu:** 09.00–10.30 WIB · **Tempat:** Ruang Rapat 2

## 1. Peserta

- Rina (pimpinan rapat)
- Budi, Sari, dan *Andi* (notulis)
  - tamu: tim desain
- ~~Dedi~~ berhalangan hadir

## 2. Keputusan

1. Rilis versi **2.4** dijadwalkan hari **Jumat, 30 Oktober 2026**.
2. Anggaran pemasaran: <span style="color: #C00000; font-weight: bold">Rp 42 juta</span> (naik ==15 %==).
3. Uji coba di Surabaya dan Bandung, <u>wajib</u> selesai sebelum rilis.

::: center
<span style="font-size: 14pt; color: navy">Target: 10.000 pengguna aktif di bulan pertama</span>
:::

## 3. Tindak lanjut

| No | Tugas | Penanggung jawab | Tenggat |
|:--:|:------|:-----------------|--------:|
| 1 | Finalisasi fitur ekspor PDF dan DOCX, termasuk uji di Word dan LibreOffice | Budi | 28 Okt |
| 2 | Materi kampanye media sosial | Sari | 29 Okt |
| 3 | Laporan uji coba lapangan | Andi | 30 Okt |

::: justify
Paragraf ini rata kiri-kanan dan cukup panjang untuk memenuhi beberapa baris, sehingga perataan kiri-kanan terlihat: kata-kata direnggangkan supaya setiap baris kecuali yang terakhir pas dengan lebar halaman. Karakter khusus juga diuji: “kutip”, ‘kutip tunggal’, em — dash, en – dash, €, ©, ü, é, ñ, dan bullet •.
:::

---

### Catatan teknis

Perintah yang dipakai: `arc-transcribe rapat.m4a --format docx`

```
arc-transcribe rapat.m4a --lang id --beam 5 --words --format docx,pdf,srt --out C:\Users\rizag\arcinferzero\transcripts\rapat-26-okt
```

\pagebreak

## Lampiran

#### Teks panjang tanpa spasi

Kata_sangat_panjang_tanpa_spasi_yang_harus_dipotong_per_karakter_karena_lebih_lebar_dari_satu_baris_halaman_penuh_abcdefghijklmnopqrstuvwxyz.

Baris pertama dengan dua spasi di akhir
baris kedua setelah ganti baris.<br>Baris ketiga lewat br.
)MD";

int main(int argc, char** argv) {
  if (argc < 2) { std::cerr << "usage: doc_test <out path without extension>\n"; return 1; }
  const std::string out = argv[1];
  doc::Options o;
  o.title = "Notulen Rapat";
  o.subtitle = "Tim Produk · Rilis 2.4";
  o.author = "arcinferzero";
  o.date = "26 Oktober 2026";
  o.header = "Rahasia internal";
  o.footer = "Notulen Rapat Tim Produk";
  {
    std::ofstream f(out + ".docx", std::ios::binary);
    const std::string d = doc::docx(kSample, o);
    f.write(d.data(), (std::streamsize)d.size());
  }
  {
    std::ofstream f(out + ".pdf", std::ios::binary);
    const std::string d = doc::pdf(kSample, o);
    f.write(d.data(), (std::streamsize)d.size());
  }
  for (const char* ext : {".rtf", ".doc"}) {  // (the .doc is RTF content: Word opens it as a normal document)
    std::ofstream f(out + ext, std::ios::binary);
    const std::string d = doc::rtf(kSample, o);
    f.write(d.data(), (std::streamsize)d.size());
  }
  doc::Options l = o;
  l.page = "Letter";
  l.landscape = true;
  l.font = "Times New Roman";
  l.line_spacing = 1.5;
  l.heading_color = "7030A0";
  l.accent_color = "FCE4D6";
  {
    std::ofstream f(out + "-letter.pdf", std::ios::binary);
    const std::string d = doc::pdf(kSample, l);
    f.write(d.data(), (std::streamsize)d.size());
  }
  {
    std::ofstream f(out + "-letter.docx", std::ios::binary);
    const std::string d = doc::docx(kSample, l);
    f.write(d.data(), (std::streamsize)d.size());
  }
  std::cout << "wrote " << out << ".docx / .pdf / -letter.pdf / -letter.docx\n";
  return 0;
}
