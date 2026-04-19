"""Generate audiobooks from .txt or .epub using Kokoro TTS on GPU.

Usage:
  python audiobook.py book.txt
  python audiobook.py book.epub --voice af_heart --speed 1.0 --out out/
  python audiobook.py book.txt --mp3     # also encode MP3 per chapter (needs ffmpeg)

List voices:
  python audiobook.py --list-voices
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLE_RATE = 24000

VOICES = {
    "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
          "am_adam", "am_michael", "am_onyx", "am_echo", "am_fenrir"],
    "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
}


def list_voices() -> None:
    print("American English (lang_code='a'):")
    for v in VOICES["a"]:
        print(f"  {v}")
    print("\nBritish English (lang_code='b'):")
    for v in VOICES["b"]:
        print(f"  {v}")
    print("\nMore voices (JP/ZH/ES/FR/HI/IT/PT) exist — see https://huggingface.co/hexgrad/Kokoro-82M")


CHAPTER_RE = re.compile(r"^\s*(chapter|part|book)\s+[\dIVXLCM]+\b.*$", re.IGNORECASE | re.MULTILINE)


def unwrap_hard_wraps(text: str) -> str:
    """Collapse single newlines inside paragraphs to spaces; preserve blank-line breaks.

    Hard-wrapped .txt (Project Gutenberg, OCR dumps) breaks lines at ~70 chars regardless
    of sentence structure, causing Kokoro to insert a pause at every line end.
    """
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" +", " ", text)
    return text


def read_txt(path: Path, unwrap: bool = True) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if unwrap:
        text = unwrap_hard_wraps(text)
    matches = list(CHAPTER_RE.finditer(text))
    if not matches:
        return [(path.stem, text.strip())]
    chapters = []
    if matches[0].start() > 50:
        chapters.append(("00_intro", text[: matches[0].start()].strip()))
    for i, m in enumerate(matches):
        title = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chapters.append((safe_name(f"{i+1:02d}_{title}"), f"{title}.\n\n{body}"))
    return chapters


def read_epub(path: Path) -> list[tuple[str, str]]:
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(path))
    chapters = []
    for idx, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        title_tag = soup.find(["h1", "h2", "h3", "title"])
        title = title_tag.get_text(strip=True) if title_tag else f"Section {idx+1}"
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 200:
            continue
        chapters.append((safe_name(f"{idx+1:02d}_{title}"), text))
    return chapters


def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().replace(" ", "_")
    return s[:80] or "chapter"


def synthesize_chapter(pipe, text: str, voice: str, speed: float) -> np.ndarray:
    parts = []
    for _gs, _ps, audio in pipe(text, voice=voice, speed=speed):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        parts.append(audio.astype(np.float32))
        parts.append(np.zeros(int(SAMPLE_RATE * 0.15), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def encode_mp3(wav_path: Path) -> Path | None:
    if shutil.which("ffmpeg") is None:
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", "96k", str(mp3_path)],
        check=True,
    )
    return mp3_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Kokoro TTS audiobook generator")
    ap.add_argument("input", nargs="?", help=".txt or .epub file")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lang", default=None, help="Kokoro lang code (a=US, b=UK, j=JP, z=ZH, e=ES, f=FR, h=HI, i=IT, p=PT). Auto from voice prefix if omitted.")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--mp3", action="store_true", help="also encode MP3 per chapter (needs ffmpeg)")
    ap.add_argument("--no-unwrap", action="store_true",
                    help="(.txt only) preserve line breaks instead of unwrapping hard-wrapped text; use for poetry/scripts")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--list-voices", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        list_voices()
        return 0
    if not args.input:
        ap.error("input file required (or pass --list-voices)")

    path = Path(args.input)
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1

    lang = args.lang or args.voice[0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Kokoro on {args.device} (lang={lang}, voice={args.voice}, speed={args.speed})...")
    t0 = time.time()
    import torch
    from kokoro import KPipeline
    if args.device == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA requested but not available — falling back to CPU", file=sys.stderr)
        args.device = "cpu"
    pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device=args.device)
    print(f"  ready in {time.time()-t0:.1f}s")

    print(f"parsing {path.name}...")
    if path.suffix.lower() == ".epub":
        chapters = read_epub(path)
    else:
        chapters = read_txt(path, unwrap=not args.no_unwrap)
    print(f"  {len(chapters)} chapter(s)")

    total_audio = 0.0
    total_wall = 0.0
    for i, (name, text) in enumerate(chapters, 1):
        print(f"[{i}/{len(chapters)}] {name}  ({len(text):,} chars)")
        t0 = time.time()
        audio = synthesize_chapter(pipe, text, args.voice, args.speed)
        dt = time.time() - t0
        dur = len(audio) / SAMPLE_RATE
        total_audio += dur
        total_wall += dt
        wav_path = out_dir / f"{name}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"    {dur/60:.1f} min audio in {dt:.1f}s  ({dur/dt if dt else 0:.1f}x realtime)  -> {wav_path.name}")
        if args.mp3:
            mp3 = encode_mp3(wav_path)
            if mp3:
                wav_path.unlink()
                print(f"    -> {mp3.name}")
            else:
                print("    (ffmpeg not found, keeping WAV)")

    print(f"\ndone: {total_audio/60:.1f} min of audio in {total_wall/60:.1f} min wall  "
          f"({total_audio/total_wall if total_wall else 0:.1f}x realtime)")
    print(f"output: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
