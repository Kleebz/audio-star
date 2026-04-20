"""Generate audiobooks from .txt or .epub using Kokoro TTS on GPU.

Usage:
  python audiobook.py book.txt                       # single book.m4b (default)
  python audiobook.py book.txt --format mp3          # single book.mp3
  python audiobook.py book.txt --format mp3 --split  # per-chapter MP3s
  python audiobook.py book.txt --format wav --split  # per-chapter WAVs (no ffmpeg needed)

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
    s = re.sub(r"\s+", " ", s).strip()
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


def chapter_display_title(name: str) -> str:
    """Turn a file-safe chapter name like '01_Chapter_1' into 'Chapter 1' for metadata."""
    return re.sub(r"^\d+_", "", name).replace("_", " ")


def run_ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def run_ffmpeg_with_progress(args: list[str], total_seconds: float) -> None:
    """Run ffmpeg and stream a live progress line based on -progress output."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)
    printed = False
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("out_time_ms="):
                continue
            try:
                secs = int(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                continue
            pct = min(100.0, 100.0 * secs / total_seconds) if total_seconds else 0.0
            print(f"\r    {pct:5.1f}%  ({secs/60:.1f} / {total_seconds/60:.1f} min)",
                  end="", flush=True)
            printed = True
    finally:
        rc = proc.wait()
        if printed:
            if rc == 0:
                print(f"\r    100.0%  ({total_seconds/60:.1f} / {total_seconds/60:.1f} min)")
            else:
                print()
        if rc != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise subprocess.CalledProcessError(rc, cmd, stderr=stderr)


def encode_chapter(wav_path: Path, fmt: str, out_dir: Path) -> Path:
    """Encode a single chapter WAV into the requested format under out_dir."""
    out_path = out_dir / f"{wav_path.stem}.{fmt}"
    if fmt == "wav":
        shutil.move(str(wav_path), str(out_path))
    elif fmt == "mp3":
        run_ffmpeg(["-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "96k", str(out_path)])
    elif fmt == "m4b":
        run_ffmpeg(["-i", str(wav_path), "-codec:a", "aac", "-b:a", "96k", str(out_path)])
    return out_path


def build_chapter_metadata(titles: list[str], durations: list[float]) -> str:
    """Build an ffmpeg FFMETADATA1 file with chapter markers in ms."""
    lines = [";FFMETADATA1"]
    start_ms = 0
    for title, dur in zip(titles, durations):
        end_ms = start_ms + int(dur * 1000)
        safe_title = (title.replace("\\", "\\\\").replace("=", "\\=")
                      .replace(";", "\\;").replace("#", "\\#").replace("\n", " "))
        lines.extend(["", "[CHAPTER]", "TIMEBASE=1/1000",
                      f"START={start_ms}", f"END={end_ms}", f"title={safe_title}"])
        start_ms = end_ms
    return "\n".join(lines) + "\n"


def concat_chapters(wav_paths: list[Path], titles: list[str], durations: list[float],
                    fmt: str, out_path: Path, tmp_dir: Path) -> None:
    """Concatenate chapter WAVs into a single output file (with chapter markers for m4b)."""
    concat_file = tmp_dir / "concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for p in wav_paths:
            abs_path = p.resolve().as_posix().replace("'", r"'\''")
            f.write(f"file '{abs_path}'\n")

    total_seconds = sum(durations)
    base_args = ["-f", "concat", "-safe", "0", "-i", str(concat_file)]
    if fmt == "m4b":
        meta_file = tmp_dir / "chapters.txt"
        meta_file.write_text(build_chapter_metadata(titles, durations), encoding="utf-8")
        run_ffmpeg_with_progress(
            base_args + ["-i", str(meta_file), "-map_metadata", "1",
                         "-codec:a", "aac", "-b:a", "96k", str(out_path)],
            total_seconds,
        )
    elif fmt == "mp3":
        run_ffmpeg_with_progress(
            base_args + ["-codec:a", "libmp3lame", "-b:a", "96k", str(out_path)],
            total_seconds,
        )
    elif fmt == "wav":
        run_ffmpeg_with_progress(
            base_args + ["-codec:a", "pcm_s16le", str(out_path)],
            total_seconds,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Kokoro TTS audiobook generator")
    ap.add_argument("input", nargs="?", help=".txt or .epub file")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--lang", default=None,
                    help="Kokoro lang code (a=US, b=UK, j=JP, z=ZH, e=ES, f=FR, h=HI, i=IT, p=PT). Auto from voice prefix if omitted.")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--format", default="m4b", choices=["m4b", "mp3", "wav"],
                    help="output format (default: m4b — the standard audiobook format)")
    ap.add_argument("--split", action="store_true",
                    help="produce per-chapter files instead of one combined file")
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

    needs_ffmpeg = args.format != "wav" or not args.split
    if needs_ffmpeg and shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH.", file=sys.stderr)
        print("install ffmpeg (https://ffmpeg.org or `winget install ffmpeg`),", file=sys.stderr)
        print("or pass `--format wav --split` for raw per-chapter WAV output.", file=sys.stderr)
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
    if not chapters:
        print("no chapters found", file=sys.stderr)
        return 1
    print(f"  {len(chapters)} chapter(s)")

    total_audio = 0.0
    total_wall = 0.0
    book_name = safe_name(path.stem)

    chapters_dir = out_dir / f"_chapters_{book_name}"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    wav_paths: list[Path] = []
    durations: list[float] = []
    titles: list[str] = []

    for i, (name, text) in enumerate(chapters, 1):
        wav_path = chapters_dir / f"{name}.wav"
        if wav_path.exists():
            dur = sf.info(str(wav_path)).duration
            total_audio += dur
            wav_paths.append(wav_path)
            durations.append(dur)
            titles.append(chapter_display_title(name))
            print(f"[{i}/{len(chapters)}] {name}  (cached, {dur/60:.1f} min)")
            continue

        print(f"[{i}/{len(chapters)}] {name}  ({len(text):,} chars)")
        t0 = time.time()
        audio = synthesize_chapter(pipe, text, args.voice, args.speed)
        dt = time.time() - t0
        dur = len(audio) / SAMPLE_RATE
        total_audio += dur
        total_wall += dt
        sf.write(wav_path, audio, SAMPLE_RATE)
        wav_paths.append(wav_path)
        durations.append(dur)
        titles.append(chapter_display_title(name))
        print(f"    {dur/60:.1f} min audio in {dt:.1f}s  ({dur/dt if dt else 0:.1f}x realtime)")

    print(f"\nencoding {args.format} output...")
    if args.split:
        for wav_path in wav_paths:
            out_path = encode_chapter(wav_path, args.format, out_dir)
            print(f"  -> {out_path.name}")
    else:
        out_path = out_dir / f"{book_name}.{args.format}"
        concat_chapters(wav_paths, titles, durations, args.format, out_path, chapters_dir)
        print(f"  -> {out_path.name}")

    shutil.rmtree(chapters_dir)

    print(f"\ndone: {total_audio/60:.1f} min of audio in {total_wall/60:.1f} min wall  "
          f"({total_audio/total_wall if total_wall else 0:.1f}x realtime)")
    print(f"output: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
