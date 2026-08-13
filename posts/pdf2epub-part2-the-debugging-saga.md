# 🕵️ pdf2epub, Part 2: The Debugging Saga — Teaching OCR to Fix Itself

*Part 2 of the `pdf2epub` series. [Part 1](pdf2epub-part1-building-the-pipeline.md) covered getting a structurally correct EPUB out of the pipeline — right cover, right chapters, no noise pages. This post is the long middle of the project: three attempts at fixing individual wrong words, a word-alignment corrector's hardest bugs, an OCR engine comparison that went nowhere on its own terms, and the real bug it accidentally exposed.*

Structurally, the EPUB was correct. But reading through it, individual words were wrong in ways that hurt comprehension — `اقرب` where it should have been `اقترب`, `الساء` where it should have been `السماء`. Both are real words in isolation, which is exactly what makes this hard: a naive "is this a valid word" check doesn't catch the first at all, and even flagging the second doesn't tell you what it should be.

---

## 🩹 Three Ways to "Fix" OCR Errors

**Auto-correct, rejected before it was built.** The simplest instinct — detect likely-wrong words and fix them automatically, no human in the loop — fails for a reason that has nothing to do with accuracy. A tool that's wrong even occasionally produces a worse failure mode than doing nothing: a *confidently* wrong word is harder to catch on a read-through than an *obviously* garbled one, because it doesn't look broken. Silent, unverifiable changes to text you can't easily diff against the original scan isn't a tradeoff worth making for a homelab tool with no review step built in.

**Context-aware outlier detection, rejected after testing.** Next idea: [AraBERT](https://github.com/aub-mind/arabert), a masked-language-model, scoring each word's likelihood given its surrounding context, flagging statistical outliers. Genuinely reasonable on paper — OCR garbling tends to produce words that don't fit context. It worked, in that it flagged real errors. It also flagged proper nouns and short function words spelled *correctly* — names and grammatical particles are inherently less predictable to a language model than ordinary body text, which is exactly the signal a z-score threshold picks up as "surprising." Tested against real content, that produced enough false positives to be actively misleading: a tool that flags true errors and correct proper nouns with equal confidence teaches you not to trust its flags at all. Raising the threshold reduced false positives but missed real errors — the fundamental problem doesn't go away with a tuning knob.

**Dictionary-based flagging, shipped.** Much less clever, works better for it: a dictionary spell-checker (`pyspellchecker`) that **flags** likely-wrong words in a review report and never touches the text. The only real enhancement over an out-of-the-box check was prefix-stripping — Arabic single-letter prepositions (و / ف / ب / ك / ل) attach directly to the following word with no space, so `والكتاب` needs its `و` stripped before `الكتاب` matches a dictionary lookup at all.

Three attempts, in decreasing order of cleverness and increasing order of usefulness. The lesson isn't "simple beats smart" — it's that a *review* tool and a *correction* tool have fundamentally different bars for acceptable error rate, and it's worth being honest about which one you're building before reaching for the more powerful technique.

## 🔬 A Second Text Source, and the Wrong Way to Use It

Flagging errors for manual review works, but manually reviewing 420 pages of flags is exactly the tedious work you build a tool to avoid. The real unlock came from a different angle: I got access to a second, independently-produced OCR extraction of the same book — better quality than Tesseract's, but structured completely differently (closer to one line per printed line, no reliable paragraph breaks). Obvious idea: use it to fix the words Tesseract got wrong.

The obvious first implementation was wrong. My first pass used the second text source as a **full replacement** for OCR'd body content — skip rasterize/OCR/cleanup entirely, structure the better text straight into chapters. This book's chapters open with a highlight/summary bullet list on their first page — a structural element OCR's paragraph detection kept as separate paragraphs, but the plain-line reference text had no signal to distinguish from regular body prose. Wholesale replacement merged those bullets straight into flowing paragraph text, corrupting the exact structure OCR had gotten right.

The fix wasn't a tweak — it was recognizing that "better text" and "structurally correct text" were two different sources, and conflating them was the actual bug. **OCR owns structure** — paragraph breaks, chapter-opening bullets, everything about *how* the text is organized. **The reference text only gets a vote on individual words** — pure word-level correction, layered on top of OCR's own structure, never replacing it.

```sh
./pdf2epub -lang ara -chapter-pages "9:ابن النيل" \
  -text-reference better-ocr-output.txt \
  -text-chapter-lines "68:ابن النيل" \
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

With the watermark actually gone rather than just less frequent, the pipeline had genuinely improved for the first time since the alignment corrector shipped. It also nearly triggered a full pivot away from the reference-text architecture entirely, for reasons that belong with the rest of what shipping this thing actually took.

**Next: [Part 3 — Lessons Learned and Going Public](pdf2epub-part3-shipping-it.md)**

---

## 🏷️ Tags
`#golang` `#python` `#nlp` `#ocr` `#arabic` `#difflib` `#paddleocr` `#tesseract` `#debugging` `#algorithms`
