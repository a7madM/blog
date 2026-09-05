---
title: "pdf2epub: Building a Self-Hosted PDF→Kindle Pipeline"
date: 2026-08-13
tags: ["go", "python", "ocr", "nlp", "arabic", "kindle", "epub", "opensource"]
cover:
  image: "images/pdf2epub-combined-thumbnail.png"
  alt: "pdf2epub: Building a Self-Hosted PDF→Kindle Pipeline"
  relative: false
aliases:
  - /posts/pdf2epub-part1-building-the-pipeline/
  - /posts/pdf2epub-part2-the-debugging-saga/
  - /posts/pdf2epub-part3-shipping-it/
---

*A Go CLI that turns scanned, image-only PDFs into Kindle-ready reflowable EPUBs — and everything that went sideways along the way.*

I had a real problem to solve: a scanned copy of an old Arabic memoir — 420 pages, image-only PDF, no text layer at all. I wanted to read it on a Kindle, reflowable, in Arabic, with proper chapters. Nothing off-the-shelf did that well without uploading a personal scan to some cloud OCR service, which didn't sit right with me.

So: `pdf2epub`. Go, local tools, homelab mindset — Tesseract instead of a cloud OCR API, everything running on my own machine.

---

## 🎯 The Goal

A CLI that takes a scanned PDF and produces a Kindle-friendly EPUB:

```sh
./pdf2epub -lang ara -title "عنوان الكتاب" -author "اسم المؤلف" \
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
  -chapter-pages "9:المقدمة;29:الفصل الثاني;65:الخاتمة" \
  -o mybook.epub input.pdf
```

All 14 of this book's chapters were found this way at first — page by page, by eye.

**Noise pages.** Some pages were blank, or photo plates with no real text — but a blank page's paper grain and dust don't OCR to nothing; they OCR to a scatter of stray digit and symbol garbage plausible enough, fragment by fragment, to survive naive filtering. The fix was a page-level density check rather than a paragraph-level one: after cleanup, if the fraction of non-space characters that are actual letters falls below a threshold (real prose runs 90%+; hallucinated noise falls well short), the whole page is dropped.

**The fix that didn't work.** The obvious next lever for OCR quality was image preprocessing — deskew, denoise, binarize before handing pages to Tesseract. Standard advice, easy to assume it must help. It didn't: tested against this book's actual pages, "cleaned up" images produced a measurably *higher* error rate than the raw rasterized ones — my best guess is that binarization discarded faint ink detail on aged paper that Tesseract's own internal processing handled better than a naive threshold did. The guess mattered less than the discipline that caught it: **measure before trusting**. An enhancement that looks obviously correct on paper still needs to be checked against real output before it ships, because "should help" and "does help" are different claims. This one got reverted, and the same discipline turned out to be the thread running through nearly every hard problem later in this project.

---

With cover, chapters, and noise pages handled, the pipeline produced a real, structurally correct EPUB. But structurally correct isn't the same as *textually* correct — individual words were still coming out of OCR wrong, sometimes badly enough to be unreadable. That's where things got interesting.

## 🩹 Three Ways to "Fix" OCR Errors

Structurally, the EPUB was correct. But reading through it, individual words were wrong in ways that hurt comprehension — `اقرب` where it should have been `اقترب`, `الساء` where it should have been `السماء`. Both are real words in isolation, which is exactly what makes this hard: a naive "is this a valid word" check doesn't catch the first at all, and even flagging the second doesn't tell you what it should be.

**Auto-correct, rejected before it was built.** The simplest instinct — detect likely-wrong words and fix them automatically, no human in the loop — fails for a reason that has nothing to do with accuracy. A tool that's wrong even occasionally produces a worse failure mode than doing nothing: a *confidently* wrong word is harder to catch on a read-through than an *obviously* garbled one, because it doesn't look broken. Silent, unverifiable changes to text you can't easily diff against the original scan isn't a tradeoff worth making for a homelab tool with no review step built in.

**Context-aware outlier detection, rejected after testing.** Next idea: [AraBERT](https://github.com/aub-mind/arabert), a masked-language-model, scoring each word's likelihood given its surrounding context, flagging statistical outliers. Genuinely reasonable on paper — OCR garbling tends to produce words that don't fit context. It worked, in that it flagged real errors. It also flagged proper nouns and short function words spelled *correctly* — names and grammatical particles are inherently less predictable to a language model than ordinary body text, which is exactly the signal a z-score threshold picks up as "surprising." Tested against real content, that produced enough false positives to be actively misleading: a tool that flags true errors and correct proper nouns with equal confidence teaches you not to trust its flags at all. Raising the threshold reduced false positives but missed real errors — the fundamental problem doesn't go away with a tuning knob.

**Dictionary-based flagging, shipped.** Much less clever, works better for it: a dictionary spell-checker (`pyspellchecker`) that **flags** likely-wrong words in a review report and never touches the text. The only real enhancement over an out-of-the-box check was prefix-stripping — Arabic single-letter prepositions (و / ف / ب / ك / ل) attach directly to the following word with no space, so `والكتاب` needs its `و` stripped before `الكتاب` matches a dictionary lookup at all.

Three attempts, in decreasing order of cleverness and increasing order of usefulness. The lesson isn't "simple beats smart" — it's that a *review* tool and a *correction* tool have fundamentally different bars for acceptable error rate, and it's worth being honest about which one you're building before reaching for the more powerful technique.

## 🔬 A Second Text Source, and the Wrong Way to Use It

Flagging errors for manual review works, but manually reviewing 420 pages of flags is exactly the tedious work you build a tool to avoid. The real unlock came from a different angle: I got access to a second, independently-produced OCR extraction of the same book — better quality than Tesseract's, but structured completely differently (closer to one line per printed line, no reliable paragraph breaks). Obvious idea: use it to fix the words Tesseract got wrong.

The obvious first implementation was wrong. My first pass used the second text source as a **full replacement** for OCR'd body content — skip rasterize/OCR/cleanup entirely, structure the better text straight into chapters. This book's chapters open with a highlight/summary bullet list on their first page — a structural element OCR's paragraph detection kept as separate paragraphs, but the plain-line reference text had no signal to distinguish from regular body prose. Wholesale replacement merged those bullets straight into flowing paragraph text, corrupting the exact structure OCR had gotten right.

The fix wasn't a tweak — it was recognizing that "better text" and "structurally correct text" were two different sources, and conflating them was the actual bug. **OCR owns structure** — paragraph breaks, chapter-opening bullets, everything about *how* the text is organized. **The reference text only gets a vote on individual words** — pure word-level correction, layered on top of OCR's own structure, never replacing it.

```sh
./pdf2epub -lang ara -chapter-pages "9:المقدمة" \
  -text-reference better-ocr-output.txt \
  -text-chapter-lines "68:المقدمة" \
  -o mybook.epub input.pdf
```

Each OCR paragraph gets aligned against the reference using word-level sequence matching (Python's `difflib.SequenceMatcher`, called from Go via a small subprocess wrapper). Simple to describe; getting there took eight real bugs, every one found by testing against this book's actual chapters, never a synthetic example. The first several were all variations of the same underlying problem — alignment scope too wide, or a cursor tracking the wrong position: one garbage paragraph could poison every paragraph after it in a whole-chapter alignment (fixed by aligning per-paragraph, not per-chapter); a paragraph near the start of the reference had nothing to anchor against (fixed by tracking a cursor and aligning against a bounded window around it); the cursor advanced on replace blocks that were considered but never actually applied (fixed by only advancing on genuine matches or accepted replacements); and fixing *that* left the cursor stale after consecutive bad paragraphs, which then overshot once the window widened to compensate — each fix's side effect became the next bug to chase.

The two bugs worth showing code for are the ones about matching itself, not cursor bookkeeping. `SequenceMatcher` diffs by **exact token equality** — `اقرب` and `اقترب` are just two unrelated tokens to it, no "almost the same word." A replace block spans from wherever it starts through to the next confirmed anchor, even when only one word in that span is the actual correction target:

```python
def align_replace_block(ocr_seg, ref_seg):
    k = min(len(ocr_seg), len(ref_seg))
    if k == 0:
        return None

    for ocr_slice, ref_slice in (
        (slice(len(ocr_seg) - k, None), slice(len(ref_seg) - k, None)),  # tail
        (slice(0, k), slice(0, k)),  # head
    ):
        result = list(ocr_seg)
        changed = False
        for o_idx, r_idx in zip(range(len(ocr_seg))[ocr_slice], range(len(ref_seg))[ref_slice]):
            if ocr_seg[o_idx] != ref_seg[r_idx]:
                result[o_idx] = ref_seg[r_idx]
                changed = True
        if changed:
            return result
    return None
```

For short replace segments, pair the OCR and reference word-for-word from both ends — tail first, then head — instead of trusting the block boundaries `SequenceMatcher` reported. The first version of this required *every* word pair in a block to agree before accepting *any* of them, which broke whenever one side had extra content the other didn't (a clean word correspondence followed by trailing punctuation with no counterpart) — requiring both to agree threw out the one clean fix along with the mismatch. The real fix judges each word pair independently: a word with no confident counterpart is left alone, a word with one gets corrected, regardless of its neighbors.

One more real regression turned up after all that: a correct word, `لها`, became the non-word `الها` — a false positive from an earlier version that gated substitution behind a similarity threshold (`difflib` ratio ≥ 0.6) as a hedge against overwriting a correct word with an unrelated one. After the alignment machinery above started producing real improvements, that threshold got removed entirely — any disagreement at an aligned position now takes the reference's word, no similarity gate. The structural safety net (small blocks, per-paragraph scoping, windowed cursor tracking) is what actually prevents nonsense corrections, not a per-word similarity score. A real, disclosed tradeoff, not a free improvement.

Eight bugs in, this is the piece of `pdf2epub` I'd point to as the hardest part of the project — not because any single bug was exotic, but because every fix's side effect only became visible against a real chapter, never in a hand-crafted test case.

## 🥊 OCR Engine Shootout: Tesseract vs. PaddleOCR

Shipping the corrector wasn't the end of the story. Reading a fresh generated EPUB end to end, whole stretches still felt wrong — noisier than "mostly correct, a few words fixed" should feel. Was this a regression somewhere, or had Tesseract's raw output just never been good enough?

Worth mentioning here where the reference text had actually come from: I'd opened the scanned PDF directly in Chrome's PDF viewer, and Chrome let me select and copy text out of it — out of an image-only PDF with no text layer at all. That's Chrome's on-device OCR, running locally, and it was noticeably better than Tesseract's on this book. Naturally: could I just use *that* engine for everything? No — no CLI, no API, no extractable binary, just a UI feature with no documented interface or batch hooks. Scripting an actual browser to drive it would be exactly the kind of fragile, undocumented automation that breaks the moment Chrome ships an update. Not worth it. I priced out cloud OCR too (Google Cloud Vision, AWS Textract, Azure Document Intelligence all land around a few dollars for a 420-page one-off) — cheap, but a direct conflict with the original no-cloud-dependency goal, so I kept the numbers for reference and tested a better *local* engine instead.

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) is the obvious local alternative with real Arabic support. A real side-by-side on actual pages from the book came back mixed: PaddleOCR recognized individual words somewhat more accurately, and — unlike Tesseract's Arabic-only language model — correctly detected Latin script mixed into a page. It also had real reading-order problems on this book's bullet-list chapter openings, scrambling a layout Tesseract handled correctly. Running the comparison out to full statistical confidence would have taken a while; I cut it short with what the sample already showed — neither engine was a clean win, and swapping the whole pipeline over would have meant trading one class of error for another, not eliminating errors.

That reframed the question. The useful thing wasn't "pick a winner" — it was noticing that PaddleOCR's correct Latin-script detection had exposed something Tesseract's Arabic-only model was actively *hiding*: a real, badly-OCR'd string of text on a huge fraction of the book's pages that had nothing to do with the actual content at all.

## 🕵️‍♂️ The Watermark Detective Story

Tesseract's Arabic-only language model has no Latin alphabet to match glyphs against. Feed it a page with Latin text on it and it doesn't fail cleanly — it forces every glyph shape into the closest thing its Arabic model has, producing plausible-looking but wrong Arabic characters. PaddleOCR, on the same pages, read that region correctly as Latin text: **"Converted by Tiff Combine — (no stamps are applied by registered version)."** A watermark — not from the book, from whatever unregistered TIFF-to-PDF tool had produced this particular scan, stamped onto page after page.

Tesseract, forced through its Arabic-only model, didn't produce one consistent misreading of that watermark — it produced *several* distinct strings of Arabic-glyph noise, varying page to page. That's exactly why it had survived undetected: the existing header/footer stripping worked by exact-string frequency, and a watermark that never OCRs to quite the same string twice just looks like N different one-off lines, each below the frequency threshold on its own.

**Measuring it properly** took a false start — an 84-page sample OCR'd at different rasterization settings than production (200 DPI + grayscale vs. the real 300 DPI, no grayscale) came back showing almost no repetition, which didn't match a glance through the book. Re-run at actual production settings: three distinct exact-string watermark variants, present on 34.1%, 6.1%, and 2.4% of pages respectively — a real, substantial, previously invisible problem.

**The obvious fix — fuzzy-matching candidate lines by string similarity instead of requiring an exact match — broke immediately.** A flat Levenshtein-ratio threshold (0.35) scored two completely unrelated real-shaped sentences at **0.469** similarity to each other, while the two real watermark variants scored **0.441**. The false-positive pair was *more* similar by this metric than the true positive pair — no single threshold could separate them, because the signal that actually distinguishes them (repeated many times vs. said once) isn't something a pairwise string-similarity score can see at all.

The real fix separates two different questions instead of asking one fuzzy one: **is this line suspicious at all** — a candidate must appear as an *exact* match on at least 3 pages before it's eligible for anything else, which is the part that actually does the work, since a one-off sentence never gets this far regardless of how structurally similar it looks to something else — and only among lines that already passed that gate, **do two candidates describe the same underlying noise**, checked by similarity. Gate on repetition, merge on similarity, never the other way around:

```go
const (
    minRepeatsForMerge           = 3
    mergeSimilarityThreshold     = 0.4
    defaultHeaderFooterThreshold = 0.3
)
```

That brought the false positives to zero and merged all three watermark variants into one strippable pattern — but a re-run still found 5 of 84 pages leaking watermark noise, because on those pages Tesseract's reading order put the watermark last instead of first, and the strip logic only checked the header pattern against the first line and the footer pattern against the last. Checking both detected patterns against both boundary positions brought that down to 1 remaining case (a stray page-number fragment pushing the watermark into the middle of the page, outside where a position-restricted check looks at all) — a documented, accepted limitation rather than a bug still being chased.

**A self-inflicted near-miss along the way:** iterating on the failed 0.35-threshold version, I ran `git checkout -- internal/cleanup/cleanup.go` to throw away that one experiment. It reverted the *entire* file, including a completely unrelated, still-uncommitted piece of work from earlier in the same session (digit normalization) that had nothing to do with the watermark fix. Caught fast — the digit-normalization test started failing immediately — and recovered with targeted edits, not another blanket operation. The lesson: `git checkout -- <file>` reverts the whole file, not "my recent change," and any file with more than one piece of uncommitted work mixed in isn't safe to blanket-revert.

Two regression tests now guard both halves of this fix directly, so the whole thing doesn't need re-discovering by staring at real OCR output again.

---

With the watermark actually gone rather than just less frequent, the pipeline had genuinely improved for the first time since the alignment corrector shipped. It also nearly triggered a full pivot away from the reference-text architecture entirely, for reasons that belong with everything else shipping this thing actually took.

## 🔢 Digit Ghosts in Three Unicode Blocks

Old Arabic books commonly number pages with Eastern Arabic-Indic numerals (`٠١٢٣٤٥٦٧٨٩`, U+0660–0669) instead of Western digits — a straightforward mapping to normalize, and worth doing for more than cosmetics: the reference-text alignment covered earlier matches word-for-word, and `١٩٤٩` and `1949` are the same number to a human but two completely different tokens to a word-alignment algorithm.

Except there's a second, visually near-identical block — Extended Arabic-Indic / Persian numerals (`۰۱۲۳۴۵۶۷۸۹`, U+06F0–06F9) — different codepoints, close to indistinguishable from the first at a glance. The first normalization pass only handled one block; a spot-check afterward showed some digits still hadn't converted, which is how the second got found. Later, reading real paragraph content while verifying an unrelated formatting fix turned up a *third*: stray digits from the Devanagari block (U+0900–U+097F) — a completely different script, not even Arabic-adjacent — leftover page-number fragments OCR had misrecognized into the wrong script entirely and bled mid-sentence:

```
وقال لي : ६१ إن اليدين تشيران
```

Same detection technique, reused three separate times: search the whole file for the Unicode block (`grep -nP '[\x{0900}-\x{097F}]'`) rather than trying to spot individual bad characters by eye.

## 🌊 The Reference File That Kept Moving

The reference text wasn't a fixed input — it kept getting edited and re-exported throughout the project, and its line count changed at least four separate times over the course of the work. Every chapter boundary is a raw line number into that file, which means every edit silently invalidates every previously-found boundary. This produced at least one real bug: a line count that didn't match what the loader reported, off by exactly the delta introduced by an edit made after boundaries were last verified but before they were next used. The fix wasn't clever, just discipline — **always re-check the line count against the current file before trusting old boundaries**, every time, no exceptions for "I just checked this yesterday."

## 🔁 Almost Repeating an Old Mistake

Even after the watermark fix, one more read-through was still disappointing enough to reopen the "maybe we need a different approach entirely" question a second time — this time, the instinct was to skip OCR correction as a concept altogether and use the reference text, by now cleaned up and clearly higher quality on its own, as the *sole* source of truth for the book's body text. That is almost exactly the full-replacement design already tried and rejected earlier: the reference text still has no signal distinguishing this book's chapter-opening bullet lists from regular body prose, and nothing about its improved quality changes that — it's a content-quality improvement, not a structural one. Catching the repeat before implementing it, not after, was the entire value of having written that rejection down properly the first time instead of carrying it as a vague memory of "we tried something like this and it didn't work."

## ✂️ Formatting Chapters: Headlines vs. Body Text

The reference text had two structurally different things living in each chapter: a handful of short chapter-opening lines (title, subtitle, headline-style teaser bullets) followed by pages of flowing prose. Paragraph-joining treated both the same way — one undifferentiated block per chapter. Two fixes: paragraph joining now splits on blank lines in the curated text file, so a hand-edited source with blank lines between the title, subtitle, and each headline bullet produces one distinct paragraph per line instead of flattening everything into one string; and for the body, a mechanical script re-inserts breaks at real sentence boundaries, targeting 220–420 characters per paragraph (500 hard cap), explicitly protecting any line that matches a chapter-marker pattern from being merged into surrounding prose.

## 🔍 "Split Into 16 Chapters" Is Not Verification

The CLI printed `split into 16 chapter(s)` and exited `0`. That's a log line, not a check. So: unzip the actual generated `.epub`, read the real `section*.xhtml` files, sample paragraph lengths, read actual sentences — which is what caught both of the next two bugs. Neither would have been visible from stdout, a passing test suite, or a chapter count that merely looked plausible.

## 📚 The Chapter That Wasn't a Chapter

Counting the actual table of contents turned up 16 entries for a 14-chapter book. Entry ten read `الفصل وتحريم الأضراب ..` — not a chapter title. **الفصل** is a genuine homograph in Arabic: it means both "chapter" and "dismissal/separation," depending on context. Buried in a paragraph about labor-law reform was the sentence "...permitting *dismissal* and prohibiting strikes," and the source text's line-wrapping happened to split that sentence so its second half landed on its own short line starting with the word "الفصل" — exactly what the chapter-marker detector looks for: a short line, prefix-matching a marker word. The detector has no semantic understanding by design, and this was the one case in 3,700+ lines where the heuristic matched and the meaning didn't. Fix: rejoin the split sentence in the source; the marker count went from 15 down to 14, all real, and every line-number boundary derived from the file (including the end-of-content cutoff) shifted by exactly the two lines that got merged away.

While in there, the *real* chapters had their own smaller bug: the table of contents was showing the plain "الفصل الأول" ("Chapter One") marker instead of each chapter's actual descriptive title, even though the EPUB generator already had that title available and just wasn't using it for the TOC label. Both bugs came from the same fifteen minutes of actually reading the generated navigation file instead of trusting that a fixed-looking chapter count meant the table of contents was fine.

## 🧵 The Thread Running Through All of It

Looking back, one habit shows up again and again, in completely different contexts: image preprocessing that looked like it should help — measurably didn't. Context-aware error detection that was theoretically sound — flagged correctly-spelled proper nouns in practice. Every alignment bug — each fix's side effect only visible against a real chapter, never a hand-crafted test case. A "better" OCR engine that traded one class of error for another, not eliminated errors. A similarity threshold that couldn't be tuned into working, because the signal it needed wasn't expressible as a pairwise score at all. Digit normalization that looked complete twice, and wasn't, three separate times. A rejected design almost taken up again for an unrelated reason, caught only because the first rejection had been written down specifically. A clean chapter count that hid a phantom chapter and a mislabeled table of contents.

**Measure before trusting.** An idea that sounds obviously correct — denoise the image, score words against a language model, "this boundary hasn't changed," "a better engine will fix it," "it built and printed the right count" — still needs to be checked against real output before you rely on it. None of the rejected approaches in this series were bad ideas; they were untested ones, and testing against the real book is what told the difference.

## 🌍 Generalizing for Open Source

Before making the repository public, I audited the Go source for anything quietly overfit to this one book. `grep` for the book's title, author, and filename anywhere in `*.go`: zero hits. Chapter-marker defaults are generic per language (`{"Chapter", "Part"}` for English, `{"الفصل", "الباب"}` for Arabic), and this book's OCR-garbled marker variants were always supplied at runtime via `-chapter-markers`, never baked into the binary. The only book-specific thing that had ever ended up in the repo, in an entire project's worth of commits, was data: the actual extracted book text (~934KB, copyrighted, not referenced by path from any code or test) and a stray `.DS_Store`. `git rm --cached` on both, plus explicit `.gitignore` entries so neither comes back by accident.

## 🗜️ Squashing a Development Log Into "v0.01"

Twelve commits, and the message quality tells its own story: `V9`, `v8`, `V7, the best ever`. Useful for me mid-project, not something worth handing a stranger reading the repo for the first time. No remote existed yet, so squashing was risk-free — no force-push, no shared history to break. The obvious approach, `git reset --soft` to the root commit, doesn't actually get you to one commit — the root commit is still sitting there as a parent. The trick that does:

```sh
git checkout --orphan squash-tmp
git commit -m "v0.01"
git branch -D main
git branch -m main
```

`--orphan` starts a new branch with the current working tree but no parent history at all. One commit, no ancestors, named for what it actually is — a first public snapshot, not a finished 1.0.

## 🚀 It's Public

**[github.com/a7madM/pdf2epub](https://github.com/a7madM/pdf2epub)**

Go, local tools, no cloud OCR dependency, Arabic RTL support, and — after everything in this series — a pipeline that's actually been checked against a real 420-page scanned book instead of just a synthetic test fixture. If there's one thread tying this whole project together, it's that every real bug in it was eventually found the same way: by reading the actual output, not the summary of it.

---

Thanks for reading — if you're building something similar (OCR pipelines, EPUB generation, or just enjoy a good debugging story), I'd love to hear about it, or see what you do with the repo. Reach out on any of the channels below.

---

## 🏷️ Tags
`#golang` `#python` `#nlp` `#ocr` `#tesseract` `#paddleocr` `#difflib` `#epub` `#kindle` `#arabic` `#unicode` `#git` `#selfhosted` `#homelab` `#cli` `#opensource` `#softwareengineering` `#lessonslearned` `#algorithms`
