"""Tests for audiobook.py text processing logic.

Only the pure-Python bits are tested here — chapter detection, filename
sanitizing, and hard-wrap unwrapping. Kokoro model inference is not tested
(it's a ~330 MB download and the model is not ours to validate).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from audiobook import CHAPTER_RE, read_epub, read_txt, safe_name, unwrap_hard_wraps


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

    def test_intro_text_before_first_chapter_is_captured(self, tmp_path):
        f = tmp_path / "book.txt"
        intro = "This is introductory material that precedes any chapter heading and is long enough to exceed the 50-character threshold in read_txt."
        f.write_text(
            f"{intro}\n\nChapter 1\n\nFirst chapter body.",
            encoding="utf-8",
        )
        chapters = read_txt(f, unwrap=False)
        assert len(chapters) == 2
        assert chapters[0][0] == "00_intro"
        assert "introductory material" in chapters[0][1]


def _build_epub(path, sections):
    """Build a minimal EPUB with the given sections.

    sections is a list of (filename, html_body_content) tuples. The body content
    is wrapped in a bare html/body shell so the `<title>` tag precedence isn't
    accidentally triggered by ebooklib-added metadata.
    """
    from ebooklib import epub as _epub

    book = _epub.EpubBook()
    book.set_identifier("id-test")
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Tester")

    items = []
    for fname, body in sections:
        c = _epub.EpubHtml(title=fname, file_name=fname, lang="en")
        c.content = f"<html><body>{body}</body></html>"
        book.add_item(c)
        items.append(c)

    book.toc = items
    book.add_item(_epub.EpubNcx())
    book.add_item(_epub.EpubNav())
    book.spine = ["nav"] + items

    _epub.write_epub(str(path), book)


class TestReadEpub:
    LONG_BODY = "<p>" + ("Some real content here. " * 20) + "</p>"  # ~500 chars

    def test_extracts_multiple_chapters(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("ch1.xhtml", f"<h1>Chapter One</h1>{self.LONG_BODY}"),
            ("ch2.xhtml", f"<h1>Chapter Two</h1>{self.LONG_BODY}"),
        ])
        chapters = read_epub(epub_path)
        assert len(chapters) == 2

    def test_filters_tiny_sections(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("cover.xhtml", "<p>Cover</p>"),  # <200 chars → filtered
            ("ch1.xhtml", f"<h1>Real Chapter</h1>{self.LONG_BODY}"),
        ])
        chapters = read_epub(epub_path)
        assert len(chapters) == 1
        assert "Real_Chapter" in chapters[0][0]

    def test_uses_heading_as_title(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("ch1.xhtml", f"<h2>The Real Title</h2>{self.LONG_BODY}"),
        ])
        chapters = read_epub(epub_path)
        assert "The_Real_Title" in chapters[0][0]

    def test_fallback_title_when_no_heading(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("ch1.xhtml", self.LONG_BODY),  # no h1/h2/h3
        ])
        chapters = read_epub(epub_path)
        assert "Section" in chapters[0][0]

    def test_body_text_is_extracted(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("ch1.xhtml",
             f"<h1>Chapter One</h1><p>The quick brown fox jumps over the lazy dog. {'x' * 300}</p>"),
        ])
        chapters = read_epub(epub_path)
        assert len(chapters) == 1
        _, body = chapters[0]
        assert "quick brown fox" in body

    def test_script_and_style_are_stripped(self, tmp_path):
        epub_path = tmp_path / "test.epub"
        _build_epub(epub_path, [
            ("ch1.xhtml",
             "<h1>Chapter One</h1>"
             "<script>alert('evil');</script>"
             "<style>body { color: red; }</style>"
             f"<p>The actual chapter text. {'x' * 300}</p>"),
        ])
        chapters = read_epub(epub_path)
        _, body = chapters[0]
        assert "actual chapter text" in body
        assert "alert" not in body
        assert "color: red" not in body
