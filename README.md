# audio-star

[![tests](https://github.com/Kleebz/audio-star/actions/workflows/tests.yml/badge.svg)](https://github.com/Kleebz/audio-star/actions/workflows/tests.yml)

A thin CLI wrapper that turns `.txt` or `.epub` books into audiobooks using [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) on your own GPU. Produces a single `.m4b` with embedded chapter markers by default, or per-chapter MP3/WAV on request.

**All the heavy lifting is done by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), an open-weight TTS model by [hexgrad](https://huggingface.co/hexgrad). This repo is just the audiobook glue — chapter detection, text cleanup, batch rendering, output formats.** See [Credits](#credits) below.

- Runs locally — nothing leaves your machine.
- Runs on CPU or GPU. GPU is typically 10–50× faster than CPU depending on hardware. On a mid-range GPU like a GTX 1080 Ti you get ~20× realtime (a 10-hour book renders in ~30 minutes); on a modern desktop CPU, expect 2–4× realtime.
- Kokoro-82M is small (~330 MB).
- No voice cloning — uses Kokoro's built-in voices (English, Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese).

**Hear it:** [`samples/demo_bella.mp3`](samples/demo_bella.mp3) — a 6-second clip using the `af_bella` voice.

## Requirements

- Python 3.10+ (tested on 3.11)
- An NVIDIA GPU is recommended but not required. CPU works — see speed notes above.
- `ffmpeg` on PATH for the default m4b and for MP3 output. Get it from [ffmpeg.org](https://ffmpeg.org/download.html) or `winget install ffmpeg`. Only `--format wav --split` works without ffmpeg.
- `espeak-ng` is **not** required — it's bundled via the `espeakng-loader` pip package.

## Install (Windows)

```cmd
git clone https://github.com/Kleebz/audio-star.git
cd audio-star

python -m venv venv
venv\Scripts\activate

:: PyTorch with CUDA — pick the right wheel for YOUR GPU + driver.
:: Visit https://pytorch.org/get-started/locally/ and copy the command,
:: or use cu121 as a safe default for most modern NVIDIA cards.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

First run also downloads the Kokoro model (~330 MB) into your Hugging Face cache (`C:\Users\<you>\.cache\huggingface\`). Subsequent runs are cached.

### CPU-only install

Skip the CUDA step, use:

```cmd
pip install torch torchaudio
```

Then pass `--device cpu` when running (the script auto-falls-back if CUDA isn't available).

## Quick start

```cmd
venv\Scripts\activate
python audiobook.py frankenstein_short.txt
```

Outputs `out/frankenstein_short.m4b` — a single audiobook file with embedded chapter markers, playable in VLC, Smart Audiobook Player, BookPlayer, Prologue, Audiobookshelf, or any audiobook app. For a longer end-to-end test, use `frankenstein_full.txt` — the full novel, public domain, produces ~4 hours of audio.

## Usage examples

```cmd
:: basic — default voice (af_heart), writes a single mybook.m4b with chapter markers
python audiobook.py mybook.txt

:: EPUB input (chapter structure is used automatically)
python audiobook.py mybook.epub

:: pick a different voice and output directory
python audiobook.py mybook.txt --voice bm_george --out audiobooks/my-book/

:: per-chapter MP3s instead of a single m4b
python audiobook.py mybook.txt --format mp3 --split

:: single combined MP3 (one file, but no chapter navigation — m4b is better for that)
python audiobook.py mybook.txt --format mp3

:: per-chapter WAVs — the only option that doesn't need ffmpeg
python audiobook.py mybook.txt --format wav --split

:: slow down slightly (default 1.0)
python audiobook.py mybook.txt --speed 0.95

:: preserve line breaks — use for poetry or scripts
python audiobook.py poetry.txt --no-unwrap

:: list available voices
python audiobook.py --list-voices
```

## Voices

Run `python audiobook.py --list-voices` to see them. Some highlights:

| Voice | Description |
|---|---|
| `af_heart` | American female, warm (default) |
| `af_bella` | American female, clear |
| `am_michael` | American male |
| `am_onyx` | American male, deeper |
| `bf_emma` | British female |
| `bm_george` | British male |

More voices (JP/ZH/ES/FR/HI/IT/PT) listed on the [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M).

## Notes

- **Hard-wrapped `.txt`** (Project Gutenberg, OCR output): auto-unwrapped by default. Mid-sentence line breaks get collapsed so Kokoro doesn't pause at each line. Use `--no-unwrap` to disable.
- **EPUB**: chapter structure is read from the embedded HTML — sections shorter than 200 chars (cover pages, blank TOCs) are skipped.
- **Chapter detection in `.txt`**: regex-based, matches lines starting with `Chapter`, `Part`, or `Book` followed by an arabic or Roman numeral. If none match, the whole file becomes a single chapter.
- **PDFs are not supported.** Convert them to EPUB or TXT first with [Calibre](https://calibre-ebook.com/).

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `input` | — | `.txt` or `.epub` file |
| `--voice` | `af_heart` | Voice name |
| `--lang` | auto | Kokoro lang code (`a`=US, `b`=UK, `j`=JP, `z`=ZH, `e`=ES, `f`=FR, `h`=HI, `i`=IT, `p`=PT). Inferred from voice prefix if omitted. |
| `--speed` | `1.0` | Playback rate multiplier |
| `--out` | `out` | Output directory |
| `--format` | `m4b` | Output format: `m4b`, `mp3`, or `wav` |
| `--split` | off | Produce per-chapter files instead of one combined file |
| `--no-unwrap` | off | `.txt` only — preserve line breaks (poetry/scripts) |
| `--device` | `cuda` | `cuda` or `cpu`; auto-falls-back if CUDA unavailable |
| `--list-voices` | — | List voices and exit |

### Output format notes

- **`m4b` (default, combined):** single `.m4b` with embedded chapter markers. The standard audiobook format — works in VLC, Smart Audiobook Player, Apple Books, BookPlayer, Prologue, Audiobookshelf, most car stereos.
- **`mp3 --split`:** folder of per-chapter MP3s. Navigation comes from track boundaries; works well in apps that treat a folder as one book.
- **`mp3` (combined):** single large MP3. MP3 chapter markers exist (ID3v2 CHAP frames) but player support is inconsistent — most audiobook apps don't read them. Use `mp3 --split` or `m4b` if you need chapter navigation.
- **`wav --split`:** raw per-chapter WAVs. The only option that works without ffmpeg. Useful if you want to edit the audio or feed it into another tool.
- **`wav` (combined):** hits the 4GB WAV format limit at roughly 11 hours of audio. ffmpeg will error past that — use `m4b` or split output for long books.

## Tests

Tests cover the text-processing logic (chapter detection, filename sanitizing, hard-wrap unwrapping). Kokoro inference itself is not tested — the model isn't ours to validate.

```cmd
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Credits

This project is a thin wrapper around work done by others. The real credit goes to:

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by [hexgrad](https://huggingface.co/hexgrad) — the TTS model. All the voice quality you hear comes from them. Released under Apache 2.0.
- **[misaki](https://github.com/hexgrad/misaki)** — the G2P (grapheme-to-phoneme) library Kokoro uses, also by hexgrad.
- **[espeak-ng](https://github.com/espeak-ng/espeak-ng)** — fallback phonemizer for out-of-dictionary words, bundled via the [`espeakng-loader`](https://pypi.org/project/espeakng-loader/) package.
- **[PyTorch](https://pytorch.org/)** — runs the model on GPU.
- **Project Gutenberg** — public-domain sample text (*Pride and Prejudice* by Jane Austen, *Frankenstein* by Mary Shelley).

If this project is useful to you, consider starring the [Kokoro model repo on Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M) — that's where the work that matters lives.

## License

[MIT](LICENSE).
