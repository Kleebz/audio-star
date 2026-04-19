# audio-star

Turn `.txt` or `.epub` books into audiobooks using [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) on your own GPU. Produces one audio file per chapter.

- Runs locally — nothing leaves your machine.
- Kokoro-82M is small (~330 MB) and fast. On a GTX 1080 Ti you get ~20× realtime: a 10-hour book renders in ~30 minutes.
- No voice cloning — uses Kokoro's built-in voices (English, Japanese, Chinese, Spanish, French, Hindi, Italian, Portuguese).

## Requirements

- Python 3.10+ (tested on 3.11)
- An NVIDIA GPU for sensible speed. CPU works but is ~10× slower.
- `ffmpeg` on PATH if you want MP3 output. Get it from [ffmpeg.org](https://ffmpeg.org/download.html) or `winget install ffmpeg`.
- `espeak-ng` is **not** required — it's bundled via the `espeakng-loader` pip package.

## Install (Windows)

```cmd
git clone https://github.com/YOUR-USERNAME/audio-star.git
cd audio-star

python -m venv venv
venv\Scripts\activate

:: PyTorch with CUDA 12.1 — adjust cu121 to your CUDA if needed
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
python audiobook.py sample.txt
```

Outputs `out/01_Chapter_1.wav` etc.

## Usage examples

```cmd
:: basic — default voice (af_heart), writes WAVs to ./out
python audiobook.py mybook.txt

:: EPUB input (chapter structure is used automatically)
python audiobook.py mybook.epub

:: pick a different voice and output directory
python audiobook.py mybook.txt --voice bm_george --out audiobooks/my-book/

:: MP3 output instead of WAV (needs ffmpeg)
python audiobook.py mybook.txt --mp3

:: slow down slightly (default 1.0)
python audiobook.py mybook.txt --speed 0.95

:: preserve line breaks — use for poetry or scripts
python audiobook.py poetry.txt --no-unwrap

:: list available voices
python audiobook.py --list-voices

:: skip venv activation — use the batch wrapper
kokoro-tts.bat mybook.txt --mp3
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
| `--mp3` | off | Encode MP3 instead of WAV (needs ffmpeg) |
| `--no-unwrap` | off | `.txt` only — preserve line breaks (poetry/scripts) |
| `--device` | `cuda` | `cuda` or `cpu`; auto-falls-back if CUDA unavailable |
| `--list-voices` | — | List voices and exit |
