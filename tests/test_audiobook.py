"""Tests for audiobook.py text processing logic.

Only the pure-Python bits are tested here — chapter detection, filename
sanitizing, and hard-wrap unwrapping. Kokoro model inference is not tested
(it's a ~330 MB download and the model is not ours to validate).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from audiobook import CHAPTER_RE, read_txt, safe_name, unwrap_hard_wraps


class TestSafeName:
    def test_spaces_become_underscores(self):
        assert safe_name("Chapter 1 Beginning") == "Chapter_1_Beginning"

    def test_strips_special_chars(self):
        assert safe_name("Ch. 1: The End!") == "Ch_1_The_End"

    def test_empty_falls_back_to_chapter(self):
        assert safe_name("") == "chapter"

    def test_truncates_to_80_chars(self):
        assert len(safe_name("x" * 200)) == 80


class TestUnwrapHardWraps:
    def test_single_newline_becomes_space(self):
        text = "one of the most\ndistinguished of that republic"
        assert unwrap_hard_wraps(text) == "one of the most distinguished of that republic"

    def test_paragraph_breaks_preserved(self):
        text = "first paragraph\n\nsecond paragraph"
        assert "\n\n" in unwrap_hard_wraps(text)

    def test_multiple_spaces_collapsed(self):
        assert unwrap_hard_wraps("foo   bar") == "foo bar"

    def test_real_gutenberg_style_passage(self):
        text = (
            "I am by birth a Genevese, and my family is one of the most\n"
            "distinguished of that republic.  My ancestors had been for many\n"
            "years counsellors and syndics."
        )
        result = unwrap_hard_wraps(text)
        assert "\n" not in result  # all line breaks collapsed
        assert "most distinguished" in result


class TestChapterRegex:
    def test_matches_arabic(self):
        assert CHAPTER_RE.search("Chapter 1: The Beginning")
        assert CHAPTER_RE.search("CHAPTER 99")

    def test_case_insensitive(self):
        assert CHAPTER_RE.search("chapter 2")

    def test_matches_roman(self):
        assert CHAPTER_RE.search("Chapter IV")
        assert CHAPTER_RE.search("Part XII")

    def test_matches_part_and_book(self):
        assert CHAPTER_RE.search("Part 1")
        assert CHAPTER_RE.search("Book III")

    def test_rejects_spelled_out_numbers(self):
        assert not CHAPTER_RE.search("Chapter One")

    def test_rejects_inline_mention(self):
        assert not CHAPTER_RE.search("as mentioned in chapter 1, the protagonist...")


class TestReadTxt:
    def test_detects_multiple_chapters(self, tmp_path):
        f = tmp_path / "book.txt"
        f.write_text(
            "Chapter 1: First\n\nSome content here.\n\n"
            "Chapter 2: Second\n\nMore content here.",
            encoding="utf-8",
        )
        chapters = read_txt(f, unwrap=False)
        assert len(chapters) == 2

    def test_no_chapters_returns_single(self, tmp_path):
        f = tmp_path / "book.txt"
        f.write_text("Just some prose with no chapter markers.", encoding="utf-8")
        chapters = read_txt(f, unwrap=False)
        assert len(chapters) == 1
        assert chapters[0][0] == "book"  # filename stem

    def test_unwrap_applied_to_chapter_body(self, tmp_path):
        f = tmp_path / "book.txt"
        f.write_text(
            "Chapter 1\n\nA line that is\nhard wrapped\nacross three lines.",
            encoding="utf-8",
        )
        chapters = read_txt(f, unwrap=True)
        assert len(chapters) == 1
        body = chapters[0][1]
        assert "hard wrapped across three lines" in body

    def test_unwrap_off_preserves_line_breaks(self, tmp_path):
        f = tmp_path / "book.txt"
        f.write_text(
            "Chapter 1\n\nLine one\nLine two",
            encoding="utf-8",
        )
        chapters = read_txt(f, unwrap=False)
        body = chapters[0][1]
        assert "Line one\nLine two" in body
