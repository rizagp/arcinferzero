# The website: a minimal version (plan, 2026-11-14)

The website is one file, `site/index.html`, published by GitHub Pages (`.github/workflows/pages.yml`) on every push
to `main` that touches `site/`. This page says what it must show, and where every number on it comes from.

## Who reads it, and what each needs on the first screen

| Reader | What they look for |
|---|---|
| A laptop owner (Core Ultra with Arc graphics) | that a 27B-35B model runs on their laptop, how fast, and a Windows download |
| An owner of a discrete Arc card (B580, Pro B60, Pro B70), or of several | the numbers on their card, one to four of them, at short and long prompts |
| A Linux user | that Linux works, which package to take (`inferzero-<version>-linux-x64.tar.gz`) and how to start it (`./zero-launcher.sh`) |

So the first screen (the hero) holds: one sentence on what it is, four headline numbers with their machine, the
two download buttons (Windows and Linux), and the requirements line. "How to start" is the last section, linked
from the top bar.

## Sections

| Section | Kept, shrunk or gone |
|---|---|
| Hero | kept; a "Download for Linux" button next to the Windows one; the 4-B60 tile changes to the newest run (long prompts) |
| Hardware (the 27B on every machine) | kept; each bar says its release or date; the long wall of notes under it becomes a short list; a new table: long prompts over 4 B60s |
| Against llama.cpp | kept as it is (laptop, 2026-10-25) |
| Results (laptop table, one-B60 table) | kept; the one-B60 table marked as older, with the newer one-B60 run beside it in a table of its own |
| What it does, Web UI, For agents | kept, unchanged (no measured numbers apart from the ones already sourced there) |
| Discrete Arc | gone as its own section: its notes move under Hardware, its "measure yours" line too. The XMX history (766 -> 1,522 tok/s) stays in `docs/xe2-tuning.md` |
| The notes under each chart | kept, as short lists instead of one paragraph |
| Get started | kept; the three steps name both packages and both launchers |

## Headline numbers

| Tile | Number | Source |
|---|---|---|
| Core Ultra 5 125H | Qwen3.6 35B-A3B, 8K prompt, 851 tok/s | `bench/website_bench.ps1`, commit 281b083, 2026-11-09 |
| Core Ultra 9 285H (Arc 140T) | Qwen3.8 27B, 4K prompt, 330 tok/s | a user on issue #5, release 2026.11.13.11 |
| 1x Arc Pro B60 | Qwen3.8 27B, 4K prompt, 1,522 tok/s | manoreken2 on issue #4, release 2026.11.14.09 (2026-11-14) |
| 4x Arc Pro B60 | Qwen3.6 35B-A3B, 64K prompt, 2,152 tok/s (131K 1,267, 262K 700) | manoreken2 on issue #4, release 2026.11.14.05 (2026-11-14) |

## Every number and its source

A number goes on the page only with its run: the laptop's `bench/website_bench.ps1`, or a user's posted output on
a GitHub issue. Each table or chart says the release (or commit) and the date. Numbers from different runs never
share one table row. Where a rerun has been asked for, the old row stays, with its date and a note that a newer run
is coming. For a run with three passes, the page takes the middle one, and says the spread where the passes differ by more
than a few percent. Within one table row, every number comes from one `runz` call (or one `website_bench` run).

| Number(s) | Source | Status |
|---|---|---|
| The laptop table (7 models, 8K / 32K / 131K, generation, drafting, KL) | `website_bench.ps1`, commit 281b083, 2026-11-09 (the 27B's 131K and Gemma's KL 2026-11-10) | current |
| Against llama.cpp, perplexity | the laptop, 2026-10-25 | current (no newer llama.cpp comparison) |
| Arc 140T 27B 4K 330 tok/s, generation 6.0 | issue #5, release 2026.11.13.11 | current |
| One B60: 27B 4K 1,522, 16K 1,359, generation 22.7 (a second run 1,524 / 23.3); 35B 4K 3,536, generation 73.8 | issue #4, release 2026.11.14.09, one run (attachment `20261114e_runz.txt`) | current |
| One B60 with MTP: 27B 51.6, 35B 82.4 / 111.9 | issue #4, release 2026.11.13.11 | older, kept with its release |
| One-B60 table, 8K / 32K / 131K, 5 models | `website_bench.ps1` on issue #4, commit 97d225b, 2026-11-10 | **stale**: before the weight prefetch; a rerun on 2026.11.14.12 was asked for on issue #4 |
| One B70: 27B 4K 2,105, 16K 1,906-1,930, generation 32.2, 75.2 with MTP | friskens on issue #12, release 2026.11.13.06 | **stale**: before the prefetch; a rerun on 2026.11.14.12 was asked for on issue #12 |
| 2 / 3 / 4 B60s on Windows, 27B 4K 1,597 / 1,724 / 1,888 | issue #4, releases 2026.11.13.05 and .11 | **stale**: before the prefetch; no rerun yet |
| 3 / 4 B60s, 27B generation 43 / 53 | issue #4, 2026-10-31 | **stale**: before the newer single-card defaults; no rerun yet |
| 35B over 4 B60s, 4K 4,264 | issue #4, 2026-11-02 | old; kept in the notes only, with its date |
| Long prompts over 4 B60s, the middle pass: 27B 1,258 / 970 / 640, 35B 2,152 / 1,267 / 700, Flash Next 948 / 778 / 526 (64K / 131K / 262K) | issue #4, release 2026.11.14.05, 2026-11-14 (attachment `20261114_runz.txt`; `docs/multi-gpu.md` records some passes' best instead) | current (new on the page as a table) |
| 2 B60s on Linux, sums GPU to GPU: 4K 1,630 (1,442 through host memory), generation 37.8 | scenic6942 on issue #17, 2026-11-14 | current |
| A card in an x4 slot halves the 2-card prompt (620 against 1,223) | issue #19, 2026-11-13 | current |
| DeepSeek-V4.1-Flash over 1-4 B60s: IQ2_XS generation 10.3 / 18.0 / 16.6 / 17.6 | issue #4, release 2026.11.14.09 | current |
| Flash Next over 1 / 2 / 4 B60s, 4K 665 / 1,202 / 1,491 | issue #4, 2026-11-13 | current |
| Core Ultra X7 358H (Arc B390), a fine-tune, 245-255 tok/s at 10K | a user's report, 2026-11-11 | kept in the notes, another file |

Not on the page: the B70 Linux numbers on issues #11 and #23 (chumbleyjl). They come from a downstream runtime, not
from this repository's code, and the reporter marks them as diagnostic.

## When new runs arrive

1. The one-B60 `website_bench` on 2026.11.14.12 (issue #4): replace the whole one-B60 table, all columns from that run.
2. The B70 on 2026.11.14.12 (issue #12): replace the B70 bars and note.
3. Multi-GPU 4K and generation reruns: replace the 2 / 3 / 4 bars, each with its release.

Each time, `README.md` ("Results at a glance" and "Discrete Arc GPUs") changes with it where it shows the same number.
