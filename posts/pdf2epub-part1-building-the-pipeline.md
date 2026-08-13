# 🏗️ pdf2epub, Part 1: Building a Self-Hosted PDF→Kindle Pipeline (and Surviving Real Scans)

*Part 1 of a 3-part series on building `pdf2epub`, a Go CLI that turns scanned, image-only PDFs into Kindle-ready reflowable EPUBs — and everything that went sideways along the way.*

I had a real problem to solve: a scanned copy of *كنت رئيسا لمصر* ("I Was President of Egypt"), Mohamed Naguib's memoir — 420 pages, image-only PDF, no text layer at all. I wanted to read it on a Kindle, reflowable, in Arabic, with proper chapters. Nothing off-the-shelf did that well without uploading a personal scan to some cloud OCR service, which didn't sit right with me.

So: `pdf2epub`. Go, local tools, homelab mindset — Tesseract instead of a cloud OCR API, everything running on my own machine.

---

## 🎯 The Goal

A CLI that takes a scanned PDF and produces a Kindle-friendly EPUB:

```sh
./pdf2epub -lang ara -title "كنت رئيسا لمصر" -author "محمد نجيب" \
  -o mybook.epub input.pdf
```

No cloud dependency, no per-page API cost, works offline once the tools are installed.

## 🏗️ v1 Architecture

The pipeline is a straight line, four stages:

```
PDF → rasterize → OCR → cleanup → EPUB
```

1. **Rasterize** (`pdftoppm`) — turn each PDF page into an image at a configurable DPI. Scanned books need real resolution here; too low and Tesseract's error rate climbs fast.
2. **OCR** (`tesseract`) — recognize text per page, language-aware (`-lang` maps to a Tesseract language pack — `ara` for Arabic).
3. **Cleanup** — raw OCR output is not a paragraph. This stage strips repeated running headers/footers, dehyphenates words split across a line wrap, and joins wrapped lines back into paragraphs using blank lines as boundaries.
4. **EPUB generation** ([`go-epub`](https://github.com/bmaupin/go-epub)) — assembles cleaned chapter text into a valid EPUB, with right-to-left support (`SetPpd("rtl")` plus `dir="rtl"` in the HTML) for Arabic.

Tesseract and Poppler (`pdftoppm`) both have Go bindings via cgo. I shelled out via `os/exec` instead — cgo bindings mean the binary has to be built against the exact library version installed on the machine, exactly the packaging friction I wanted to avoid for a homelab tool that should just run wherever `tesseract` and `poppler-utils` are installed via the system package manager. `os/exec` costs a bit of process-spawn overhead per page; it buys portability and a much simpler build.

Before ever touching the real 420-page book, `pdf2epub` was scaffolded and tested against small synthetic PDFs — no OCR surprises, just proving rasterize → OCR → cleanup → EPUB correct in isolation. That mattered more than it sounds: once real scanned data enters the picture, it's much easier to debug a broken pipeline stage on a 3-page synthetic file than to figure out whether a bug is in the pipeline or in the data, 200 pages into a real book.

Once tested against the real book, single-threaded rasterize→OCR was too slow to iterate on — 420 pages through Tesseract, one at a time, turns every test run into a coffee break. The fix was a bounded worker pool: a semaphore channel capped by a `-concurrency` flag (default `runtime.NumCPU()`), with a `sync.WaitGroup` to join and a small shared progress helper reporting live status across workers. That turned a full-book run from "leave it overnight" into something you can actually iterate on during a work session.

---

## 🖼️ Then the Real Book Showed Up

The pipeline worked cleanly on small synthetic test PDFs. Then I ran it against the real thing, and real scans expose problems synthetic test files never will.

**No cover.** The naive assumption — "page 1 is the cover" — was wrong; page 1 was blank front matter. Needed an explicit `-cover-page 5` flag, and this only became obvious by actually opening the generated EPUB and looking at it. No amount of code review catches "the cover is a blank page," only inspection of the real output.

**Chapters OCR couldn't see.** This book's chapter headings used a decorative font — visually obvious to a human, but Tesseract sometimes failed to recognize that text *at all*, even while reading the surrounding body text on the same page perfectly. A marker-word chapter detector (look for paragraphs starting with `"الفصل,الباب"`) simply never sees the marker if OCR never produced the word in the first place. `pdf2epub` ended up supporting two detection strategies: **automatic** (`-chapter-markers`, works when headings share the body font) and **manual** (`-chapter-pages`, for everything else — transcribe each chapter's physical PDF page number and title, usually easiest from the book's own printed table of contents, which OCRs fine even when the decorative headings don't):

```sh
./pdf2epub -lang ara \
  -chapter-pages "9:ابن النيل;29:سنوات الخدمة;65:حرب فلسطين" \
  -o mybook.epub input.pdf
```

All 14 of this book's chapters were found this way at first — page by page, by eye.

**Noise pages.** Some pages were blank, or photo plates with no real text — but a blank page's paper grain and dust don't OCR to nothing; they OCR to a scatter of stray digit and symbol garbage plausible enough, fragment by fragment, to survive naive filtering. The fix was a page-level density check rather than a paragraph-level one: after cleanup, if the fraction of non-space characters that are actual letters falls below a threshold (real prose runs 90%+; hallucinated noise falls well short), the whole page is dropped.

**The fix that didn't work.** The obvious next lever for OCR quality was image preprocessing — deskew, denoise, binarize before handing pages to Tesseract. Standard advice, easy to assume it must help. It didn't: tested against this book's actual pages, "cleaned up" images produced a measurably *higher* error rate than the raw rasterized ones — my best guess is that binarization discarded faint ink detail on aged paper that Tesseract's own internal processing handled better than a naive threshold did. The guess mattered less than the discipline that caught it: **measure before trusting**. An enhancement that looks obviously correct on paper still needs to be checked against real output before it ships, because "should help" and "does help" are different claims. This one got reverted, and the same discipline turned out to be the thread running through nearly every hard problem later in this project.

---

With cover, chapters, and noise pages handled, the pipeline produced a real, structurally correct EPUB. But structurally correct isn't the same as *textually* correct — individual words were still coming out of OCR wrong, sometimes badly enough to be unreadable. That's where things got interesting.

**Next: [Part 2 — The Debugging Saga: Teaching OCR to Fix Itself](pdf2epub-part2-the-debugging-saga.md)**

---

## 🏷️ Tags
`#golang` `#ocr` `#tesseract` `#epub` `#kindle` `#selfhosted` `#homelab` `#cli` `#arabic`
