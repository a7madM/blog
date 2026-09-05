---
title: "Building photo-dedupe: a local-first burst-photo cleaner in Go"
date: 2026-09-05
draft: true
tags: ["go", "cli", "photography"]
---

Every phone camera has the same bad habit: burst mode. You hold the shutter
for a second and get seven nearly-identical frames, and later you're the one
who has to squint at them side by side and decide which is "the good one."
`photo-dedupe` is a small Go CLI (with an optional local web UI) that does
that triage for you — clustering photos by capture time, grouping the ones
that actually look alike, and picking the sharpest, highest-resolution frame
in each group as the keeper. Nothing is ever deleted: losers are moved into
a quarantine folder for a human to review, and the whole thing runs offline
against files on disk, with no cloud calls and no accounts. What makes it
worth writing up isn't any single algorithm — perceptual hashing and
Laplacian-variance sharpness are both well-known techniques — but how
cleanly the problem decomposes into independent, testable stages, and how
much of the design is really about safety: a dry-run plan file that's just
JSON you can read, a two-step move instead of a delete, and a restore path
that reverses the whole operation.

## Article: how it was built, stage by stage

### 1. Start from the decision, not the algorithm

Before writing any code, the shape of the problem is: given a folder of
photos, find groups of near-duplicates, and within each group, pick one
winner. That's naturally three separate questions:

1. Which photos were taken *around the same time*? (a candidate pool)
2. Within that pool, which ones actually *look alike*? (real duplicates)
3. Within a group of real duplicates, which one is *best*? (the winner)

Keeping those three questions as three separate packages — rather than one
big "dedupe" function — turned out to be the single most important design
decision. Each one became independently testable with plain data in, plain
data out, no file I/O, no images involved at all for two of the three.

### 2. Time-clustering: a pure function over timestamps

The first stage doesn't look at pixels at all. It sorts capture timestamps
and splits the sequence wherever the gap between two consecutive shots
exceeds a threshold (default 60s). This is deliberately *not* a duplicate
detector — it just narrows down which photos are even worth comparing
pixel-by-pixel, since comparing every photo in a 10,000-photo library
against every other photo is quadratic and pointless.

```go
// internal/cluster/cluster.go
type Item struct {
	Path      string
	Timestamp time.Time
}

func Group(items []Item, gap time.Duration) [][]Item {
	sorted := make([]Item, len(items))
	copy(sorted, items)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})

	groups := [][]Item{{sorted[0]}}
	for _, item := range sorted[1:] {
		last := groups[len(groups)-1]
		prev := last[len(last)-1]
		if item.Timestamp.Sub(prev.Timestamp) > gap {
			groups = append(groups, []Item{item})
		} else {
			groups[len(groups)-1] = append(last, item)
		}
	}
	return groups
}
```

Because this takes plain `Item{Path, Timestamp}` structs, its test suite
never has to touch a real image file — it's pure data-in/data-out, which
made it the first package written and the fastest to get to 100% confidence.

Where do the timestamps come from? A small `exiftime` package reads EXIF
`DateTimeOriginal` and falls back to file mtime if there's no EXIF data (or
no EXIF library support for that format) — every photo has *a* timestamp,
just not always a reliable one.

### 3. Similarity grouping: union-find over an injected distance function

A time-cluster is only a *candidate pool*. Being taken 2 seconds apart never
implies duplication on its own — you could take one photo, wait, then take
an unrelated one in the same burst window. The actual "do these look alike"
decision needs a perceptual hash comparison, but the *grouping logic* itself
— i.e., connected components over a threshold — doesn't need to know
anything about image hashing:

```go
// internal/simgroup/simgroup.go
type DistanceFunc func(i, j int) int

func Group(n int, threshold int, dist DistanceFunc) [][]int {
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if dist(i, j) <= threshold {
				union(i, j)
			}
		}
	}
	// ...collect groups from parent[]
	return result
}
```

`simgroup` takes an `n` and a `DistanceFunc` — it has no idea what a photo
is. The caller (the `scan` orchestrator) supplies the real distance function,
which compares two images' perceptual hashes (via `goimagehash`) and returns
their Hamming distance:

```go
distFn := func(i, j int) int {
	d, err := groupEntries[i].metrics.Hash.Distance(groupEntries[j].metrics.Hash)
	if err != nil {
		return opts.SimilarityThreshold + 1 // treat as dissimilar
	}
	return d
}
simGroups := simgroup.Group(len(groupEntries), opts.SimilarityThreshold, distFn)
```

This injection is what let `simgroup`'s tests run against fake integer
distances instead of real decoded images — fast, deterministic, and
independent from every decode-related edge case (corrupt files, unsupported
formats, missing `magick`, etc).

### 4. Scoring an image: sharpness and perceptual hash

`imagemetrics` is the one package with no unit tests by design — sharpness
and perceptual-hash scores are validated empirically against real sample
photos, not asserted as a deterministic contract, since "is this photo
blurry" doesn't have a single correct numeric answer to assert against.

Sharpness is the variance of the image's Laplacian: convert to grayscale,
convolve with a discrete Laplacian kernel, and take the variance of the
result. A sharp image has a lot of high-frequency detail (edges), so the
Laplacian response varies a lot pixel to pixel; a blurry image is smoother,
so the variance is low.

```go
// internal/imagemetrics/imagemetrics.go
func sharpness(img image.Image) float64 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	gray := make([][]float64, h)
	for y := 0; y < h; y++ {
		gray[y] = make([]float64, w)
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			gray[y][x] = 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
		}
	}

	var sum, sumSq, n float64
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			lap := gray[y-1][x] + gray[y+1][x] + gray[y][x-1] + gray[y][x+1] - 4*gray[y][x]
			sum += lap
			sumSq += lap * lap
			n++
		}
	}
	mean := sum / n
	return sumSq/n - mean*mean
}
```

HEIC/HEIF is the awkward format here: Go's standard library and the
`goimagehash`/`image` ecosystem have no native decoder for it, and there's
no pure-Go alternative worth depending on. Rather than pull in a heavy CGO
binding, the project shells out to the system's `magick` (ImageMagick)
binary, converts to PNG on stdout, and decodes that with the stdlib:

```go
func decodeHEIC(path string) (image.Image, error) {
	cmd := exec.Command("magick", path, "png:-")
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("magick decode of %s failed: %w: %s",
			path, err, strings.TrimSpace(stderr.String()))
	}
	img, _, err := image.Decode(&stdout)
	return img, err
}
```

Importantly, a missing or HEIF-incapable `magick` never crashes a scan —
`Compute`'s caller treats a decode failure exactly like any other
unreadable file: skip it, log a warning, move on. This "degrade, don't
crash" posture shows up throughout the codebase.

### 5. Picking a winner: sharpness as a filter, not a ranking

The subtlest design decision in the whole project is in `pick`. The naive
approach is "sort by sharpness, take the sharpest." That's wrong in
practice: a tiny, heavily-compressed thumbnail can register a high Laplacian
variance from noise, while a genuinely great, full-resolution shot that's
merely *slightly* softer than the sharpest frame in the group would get
discarded. So sharpness is used as an *eligibility filter* — anything more
than `blurThreshold` below the group's best sharpness is disqualified
outright — and only among the survivors does resolution (then file size,
then path) decide the actual winner:

```go
// internal/pick/pick.go
func Pick(candidates []Candidate, blurThreshold float64) (winner Candidate, losers []Candidate) {
	maxSharpness := candidates[0].Sharpness
	for _, c := range candidates[1:] {
		if c.Sharpness > maxSharpness {
			maxSharpness = c.Sharpness
		}
	}

	winnerIdx := -1
	for i, c := range candidates {
		if maxSharpness-c.Sharpness > blurThreshold {
			continue // disqualified: too much blurrier than the group's best
		}
		if winnerIdx == -1 || better(c, candidates[winnerIdx]) {
			winnerIdx = i
		}
	}
	// ...split candidates into winner + losers
}

func better(a, b Candidate) bool {
	if a.resolution() != b.resolution() {
		return a.resolution() > b.resolution()
	}
	if a.SizeBytes != b.SizeBytes {
		return a.SizeBytes > b.SizeBytes
	}
	return a.Path < b.Path // final deterministic tiebreak
}
```

That last tiebreak — lexicographic path — matters more than it looks: it
guarantees `Pick` is fully deterministic given the same inputs, which makes
the whole pipeline reproducible and testable without any randomness to
paper over.

### 6. The plan file: a dry-run contract you can read

Everything above only *decides*; nothing touches disk. The output of a scan
is a `Plan` — a JSON document listing every group, its winner, its losers,
and (critically) a SHA-256 content hash of every file recorded at scan time:

```go
// internal/plan/plan.go
type FileRecord struct {
	Path        string  `json:"path"`
	ContentHash string  `json:"content_hash"`
	Width       int     `json:"width"`
	Height      int     `json:"height"`
	Sharpness   float64 `json:"sharpness"`
	SizeBytes   int64   `json:"size_bytes"`
}

type Group struct {
	ID     int          `json:"id"`
	Winner FileRecord   `json:"winner"`
	Losers []FileRecord `json:"losers"`
}

type Plan struct {
	Version     int       `json:"version"`
	Root        string    `json:"root"`
	GapSeconds  int       `json:"gap_seconds"`
	GeneratedAt time.Time `json:"generated_at"`
	Groups      []Group   `json:"groups"`
}
```

Making the plan a plain, human-readable JSON file (not a database, not an
opaque binary) is a deliberate trust-building choice: a cautious user can
open `.dedupe-plan.json`, read exactly what the tool intends to do, and only
then run `apply`. The content hash is what makes `apply` safe to run later,
possibly after the user has touched files in between — `apply` re-hashes
every winner and loser against the plan's recorded hash right before moving
it, and skips (rather than blindly moves) anything that's drifted.

### 7. Apply and restore: move, never delete

`apply` is the *only* package in the codebase allowed to mutate the
filesystem, and it only does two things: move winners into `dedupe-kept/`
and losers into `dedupe-quarantine/`, preserving each file's original
relative path under whichever folder it lands in. `restore` reverses this
using the same plan file. Nothing is ever hard-deleted — quarantine is a
holding area for the user to review and clear out themselves once they
trust the results.

This split (decide vs. act, as two entirely separate commands operating off
a durable file) is what makes the tool safe to experiment with: you can run
`scan` as many times as you want, inspect and re-tune `-similarity`/`-blur`,
and only run `apply` once you're confident — and even then, `restore` is one
command away.

### 8. Orchestration: wiring the stages together, concurrently

`scan.Run` is the glue: walk the directory, resolve every file's timestamp
and metrics, feed the results through `cluster` → `simgroup` → `pick`, and
assemble a `Plan`. The expensive step — decoding an image, hashing it, and
scoring its sharpness — is embarrassingly parallel (every file is
independent of every other), so it runs across a bounded worker pool sized
to `runtime.NumCPU()` by default:

```go
// internal/scan/scan.go
func resolveEntries(paths []string, concurrency int, progress func(index, total int, path string)) ([]entry, []Warning) {
	if concurrency <= 0 {
		concurrency = runtime.NumCPU()
	}

	var mu sync.Mutex
	var entries []entry
	var warnings []Warning
	done := 0

	jobs := make(chan string)
	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for path := range jobs {
				e, warn := resolveOne(path)
				mu.Lock()
				if warn != nil {
					warnings = append(warnings, *warn)
				} else {
					entries = append(entries, e)
				}
				done++
				progress(done, len(paths), path)
				mu.Unlock()
			}
		}()
	}
	for _, path := range paths {
		jobs <- path
	}
	close(jobs)
	wg.Wait()
	return entries, warnings
}
```

Clustering and grouping still happen *after* every file is resolved, so the
final result is identical regardless of how many workers ran or in what
order they finished — only the "which file finished first" ordering is
non-deterministic, not the output. That claim is backed by a
race-detector test (`go test -race`) that scans the same sample directory
sequentially and concurrently and asserts the resulting plans match. On a
150-photo sample this made scanning about 3.8× faster.

Directory discovery also enforces the project's core safety invariant:
`dedupe-kept/` and `dedupe-quarantine/` are always excluded from the walk,
which is what makes re-running `scan` on an already-applied directory safe
and idempotent.

### 9. A CLI, then a local browser UI, on the same core

`cmd/dedupe/main.go` stays intentionally thin: it just parses flags per
subcommand (`scan`, `apply`, `restore`, `serve`) and calls into the
packages above. `serve` was added later as a `net/http` handler
(`internal/webui`) wrapping the exact same `scan.Run`/`apply.Apply`/
`apply.Restore` calls — no parallel logic, no duplicated decision-making. It
binds to loopback only, holds one in-memory `Plan` behind a mutex, and
serves images by re-encoding HEIC to JPEG on the fly, but only for paths
that are actually part of the currently loaded plan — nothing else on disk
is reachable through it. That constraint (serve only what's in the plan,
never an arbitrary path off the filesystem) is the same trust boundary the
CLI has, just enforced over HTTP instead of by construction.

### 10. Why the pipeline order matters

Looking back at the whole thing end to end:

```
discover → (exiftime + imagemetrics, concurrent) → cluster → simgroup → pick → plan → apply/restore
```

Each arrow is a package boundary with a narrow, typed interface, and every
stage except `imagemetrics` (decoding real images) is pure data in, pure
data out — which is exactly what made the test suite (`go test ./...`)
cheap to write and fast to run. The lesson generalizes past this project:
when a task naturally decomposes into "narrow the candidates," "confirm the
match," and "rank what's left," keeping those as three distinct, injectable
stages — rather than one function that does clustering-and-comparison-and-
ranking together — pays for itself the first time you need to change just
one of the three thresholds without touching the other two.
