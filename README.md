# OneFill: evaluation data

Evaluation data for the paper *OneFill: one-step, text-free filling of long speech gaps* (submitted to Knowledge-Based Systems):
the test clips, per-clip scores of every system in the main comparison, and the anonymised listening-test ratings with the
analysis script. Every number of the main comparison and of the listening test can be recomputed from these files.

The code and the trained checkpoints will be released after acceptance.

## Contents

| Path | What it is |
|---|---|
| `cohorts/clip_list.csv` | The 300 test clips per gap length: clip id, utterance index in LibriTTS-R test-clean, speaker, gap start and length (ms). |
| `cohorts/ltr_test_{1000,500,200}_meta.json` | The same cohorts in the format used by the evaluation code (gap start/end in samples at 16 kHz). |
| `per_clip/OneFill_per_clip_long.csv` | Per-clip scores, one row per clip × gap length × system (12,900 rows). |
| `listening_test/responses/` | The 20 listening-test result files (identifiers and browser information removed). |
| `listening_test/ratings_long.csv` | The same ratings decoded to systems, one row per rating (1,440 rows). |
| `listening_test/stimulus_info/` | Answer key (`stim_key.json`), sentence list (`items.json`) and objective metrics on the exact audio played (`stim_metrics.json`). |
| `listening_test/analyze_v4_3.py` | The pre-specified analysis; `listen_v4_3_analysis.txt/.json` is its output. |

## Test clips

LibriTTS-R test-clean (3565 utterances of at least 2.5 s). 300 utterances are taken at evenly spaced indices
(`numpy.linspace`), each cropped to its central 2.5 s, with a centred gap of 1000, 500 or 200 ms; the same clips are used
for every gap length and every system. `utterance_index` counts the test-clean utterances of at least 2.5 s in the order of
the official parquet shards.

## Per-clip scores

Columns of `OneFill_per_clip_long.csv`:

| Column | Meaning |
|---|---|
| `utmos`, `plcmos`, `dnsmos` | whole-clip UTMOS, PLCMOS, DNSMOS OVRL |
| `wer` | Whisper-small word error rate on the whole clip |
| `spk` | ECAPA cosine similarity between the filled gap and the surrounding context |
| `utmos_g`, `dnsmos_g` | UTMOS and DNSMOS on the gap ± 100 ms |
| `per_anchor` | phone error rate on the gap ± 0.5 s, divided by the number of reference phones in that window |
| `nisqa`, `nisqa_g` | NISQA on the whole clip and on the gap ± 100 ms |

Systems: `clean` (original), `zero fill`, `OneFill` (one step), baselines (`FRN`, `Opus 1.5 PLC`, `LPC`, `Asaad et al.`,
`A2SB`, `UniPASE`, `Complex-bin2bin`, `GSLM-style units`, `cascade (ASR+LLM+TTS)`, `reduced cascade`,
`F5-TTS with transcript`) and the control `other words, same voice`.

## Listening test

MUSHRA-like page with a 5-point naturalness scale; 12 sentences (1 s gaps) × 6 versions: `ref` hidden original, `anchor`
zero fill, `gslm` GSLM-style units, `ours1` OneFill 1 step, `ours16` OneFill 16 steps, `casc` cascade (ASR + LLM + TTS).
20 listeners (Mandarin-native graduate students, informed consent); 18 kept by the pre-specified screening rule.

```bash
cd listening_test
mkdir -p inputs && cp stimulus_info/*.json inputs/
python analyze_v4_3.py inputs responses     # writes responses/listen_v4_3_analysis.txt (identical to the copy here)
```

Requirements: Python 3, numpy, scipy.

## License

CC BY 4.0. LibriTTS-R itself is distributed under its own license (CC BY 4.0) by its authors.
