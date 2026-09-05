---
title: "Building photo-dedupe: a local-first burst-photo cleaner in Go"
date: 2026-08-30
draft: false
tags: ["go", "cli", "photography", "performance"]
---

Burst mode leaves you with seven nearly-identical frames and no easy way to pick "the good one." `photo-dedupe` is a small offline Go CLI (plus an optional local web UI) that does that triage: cluster by capture time, group the shots that actually look alike, keep the sharpest and highest-resolution frame. Nothing is ever deleted — losers move to a quarantine folder for you to review. What's worth writing up isn't the algorithms, which are well-known, but how cleanly the problem splits into independent, testable stages, and how much of the design is really about safety: a readable JSON plan, a move instead of a delete, a one-command restore.

## 1. Start from the decision, not the algorithm

Before writing any code, the shape of the problem is: given a folder of photos, find groups of near-duplicates, and within each group, pick one winner. That's naturally three separate questions:

- Which photos were taken *around the same time*? (a candidate pool)
- Within that pool, which ones actually *look alike*? (real duplicates)
- Within a group of real duplicates, which one is *best*? (the winner)

Keeping those as three separate packages, not one big "dedupe" function, was the single most important design decision: each became independently testable — plain data in, plain data out, no images involved for two of the three.

## 2. Time-clustering: a pure function over timestamps

The first stage never touches pixels: it sorts capture timestamps and splits wherever the gap exceeds a threshold (default 20s). It's not a duplicate detector — just a way to avoid comparing every photo in a 10,000-photo library against every other one.

```go
// internal/cluster/cluster.go

// Group partitions timestamps (indices 0..len(timestamps)-1) into
// clusters, splitting wherever the gap between chronologically
// consecutive timestamps exceeds gap. Groups are returned in
// chronological order; each group's indices are ordered ascending by
// timestamp.
func Group(timestamps []time.Time, gap time.Duration) [][]int {
    if len(timestamps) == 0 {
        return nil
    }

    order := make([]int, len(timestamps))
    for i := range order {
        order[i] = i
    }
    sort.Slice(order, func(i, j int) bool {
        return timestamps[order[i]].Before(timestamps[order[j]])
    })

    groups := [][]int{{order[0]}}
    for _, idx := range order[1:] {
        last := groups[len(groups)-1]
        prev := last[len(last)-1]
        if timestamps[idx].Sub(timestamps[prev]) > gap {
            groups = append(groups, []int{idx})
        } else {
            groups[len(groups)-1] = append(last, idx)
        }
    }

    return groups
}
```

It takes plain `[]time.Time` and returns index groups — no `Path`, no notion of a "photo." That made it the first package written and the fastest to fully trust. Timestamps come from `exiftime`: EXIF `DateTimeOriginal`, falling back to file mtime.

## 3. Similarity grouping: union-find over an injected distance function

A time-cluster is only a candidate pool — being seconds apart never implies duplication on its own. The "do these look alike" check needs perceptual hashing, but the grouping logic itself — connected components over a threshold — doesn't need to know anything about images:

```go
// internal/simgroup/simgroup.go

// DistanceFunc returns a distance between items i and j (0..n-1).
// Implementations are expected to be symmetric: dist(i,j) == dist(j,i).
type DistanceFunc func(i, j int) int

// Group partitions n items (indices 0..n-1) into clusters where an
// edge exists between i and j whenever dist(i, j) <= threshold.
// Clusters are returned as slices of ascending indices, ordered by
// each group's smallest index — deterministic for a given n and dist,
// so callers can rely on repeat runs over the same input producing
// the same group order.
func Group(n int, threshold int, dist DistanceFunc) [][]int {
    if n == 0 {
        return nil
    }

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

    byRoot := make(map[int][]int)
    for i := 0; i < n; i++ {
        root := find(i)
        byRoot[root] = append(byRoot[root], i) // i ascends, so each group stays sorted
    }

    result := make([][]int, 0, len(byRoot))
    for _, g := range byRoot {
        result = append(result, g)
    }
    sort.Slice(result, func(i, j int) bool { return result[i][0] < result[j][0] })
    return result
}
```

`simgroup` has no idea what a photo is — the caller supplies the real distance function, comparing perceptual hashes via `goimagehash`:

```go
// internal/scan/scan.go — the injected distance function

distFn := func(i, j int) int {
    d, err := groupEntries[i].metrics.Hash.Distance(groupEntries[j].metrics.Hash)
    if err != nil {
        return opts.SimilarityThreshold + 1 // treat as dissimilar
    }
    return d
}
simGroups := simgroup.Group(len(groupEntries), opts.SimilarityThreshold, distFn)
```

That injection lets `simgroup`'s tests run on fake integer distances instead of real images — fast, deterministic, and immune to every decode edge case.

## 4. Scoring an image: sharpness and perceptual hash

`imagemetrics` has no unit tests by design — sharpness and hash scores are validated empirically against real photos, since "is this blurry" has no single correct numeric answer.

Sharpness is the variance of the image's Laplacian: a sharp image has lots of high-frequency detail, so the response varies a lot pixel to pixel; a blurry one is smoother. The first version built the grayscale buffer via `img.At(x,y).RGBA()` per pixel — correct but slow, since each call boxes a `color.Color` onto the heap. Reading decoded bytes directly for the three concrete types actually seen cut that cost roughly 10×:

```go
// internal/imagemetrics/imagemetrics.go

func sharpness(img image.Image) float64 {
    bounds := img.Bounds()
    w, h := bounds.Dx(), bounds.Dy()
    if w < 3 || h < 3 {
        return 0
    }

    gray := grayscale(img, bounds, w, h)

    var sum, sumSq float64
    var n float64
    for y := 1; y < h-1; y++ {
        row, up, down := gray[y*w:], gray[(y-1)*w:], gray[(y+1)*w:]
        for x := 1; x < w-1; x++ {
            lap := up[x] + down[x] + row[x-1] + row[x+1] - 4*row[x]
            sum += lap
            sumSq += lap * lap
            n++
        }
    }
    if n == 0 {
        return 0
    }
    mean := sum / n
    return sumSq/n - mean*mean
}

// *image.Gray and *image.RGBA are exact matches for the same weighted
// sum; *image.YCbCr (what the stdlib JPEG decoder produces, the
// dominant real-world format here) is a documented near-exact
// stand-in, since Y is already JPEG's own luma channel. Anything else
// falls back to the general img.At() path, unchanged from before.
func grayscale(img image.Image, bounds image.Rectangle, w, h int) []float64 {
    gray := make([]float64, w*h)

    switch src := img.(type) {
    case *image.Gray:
        for y := 0; y < h; y++ {
            off := src.PixOffset(bounds.Min.X, bounds.Min.Y+y)
            row := src.Pix[off : off+w]
            out := gray[y*w : y*w+w]
            for x, v := range row {
                out[x] = float64(v) * 257 // Y*0x101, matches color.Gray.RGBA()
            }
        }
    case *image.YCbCr:
        for y := 0; y < h; y++ {
            off := src.YOffset(bounds.Min.X, bounds.Min.Y+y)
            row := src.Y[off : off+w]
            out := gray[y*w : y*w+w]
            for x, v := range row {
                out[x] = float64(v) * 257
            }
        }
    case *image.RGBA:
        for y := 0; y < h; y++ {
            off := src.PixOffset(bounds.Min.X, bounds.Min.Y+y)
            row := src.Pix[off : off+4*w]
            out := gray[y*w : y*w+w]
            for x := 0; x < w; x++ {
                i := x * 4
                r, g, b := float64(row[i]), float64(row[i+1]), float64(row[i+2])
                out[x] = (0.299*r + 0.587*g + 0.114*b) * 257
            }
        }
    default:
        for y := 0; y < h; y++ {
            out := gray[y*w : y*w+w]
            for x := 0; x < w; x++ {
                r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
                out[x] = 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)
            }
        }
    }
    return gray
}
```

HEIC/HEIF has no native Go decoder and no pure-Go alternative worth depending on, so the project shells out to `magick` (ImageMagick), decoding its PNG output with the stdlib — pre-sizing the buffer off the source file's size to avoid repeated grow-and-copy:

```go
// internal/imagemetrics/imagemetrics.go

func decodeHEIC(path string) (image.Image, error) {
    cmd := exec.Command("magick", path, "png:-")
    var stdout, stderr bytes.Buffer
    if info, err := os.Stat(path); err == nil {
        stdout.Grow(int(info.Size()) * 4)
    }
    cmd.Stdout = &stdout
    cmd.Stderr = &stderr
    if err := cmd.Run(); err != nil {
        return nil, fmt.Errorf("magick decode of %s failed: %w: %s", path, err, strings.TrimSpace(stderr.String()))
    }

    img, _, err := image.Decode(&stdout)
    return img, err
}
```

A missing or HEIF-incapable `magick` never crashes a scan — the file is just skipped and logged, same as any other unreadable file.

## 5. Picking a winner: sharpness as a filter, not a ranking

The subtlest decision in the project is in `pick`. "Sort by sharpness, take the sharpest" is wrong in practice — a noisy thumbnail can register high Laplacian variance, discarding a genuinely great shot that's merely slightly softer. So sharpness is an eligibility filter, not a ranking: anything more than `blurThreshold` below the group's best is disqualified, and only among survivors does resolution, then size, then path decide the winner:

```go
// internal/pick/pick.go

func Pick(candidates []Candidate, blurThreshold float64) (winner Candidate, losers []Candidate) {
    if len(candidates) == 0 {
        return Candidate{}, nil
    }

    maxSharpness := candidates[0].Sharpness
    for _, c := range candidates[1:] {
        if c.Sharpness > maxSharpness {
            maxSharpness = c.Sharpness
        }
    }

    winnerIdx := -1
    for i, c := range candidates {
        if maxSharpness-c.Sharpness > blurThreshold {
            continue
        }
        if winnerIdx == -1 || better(c, candidates[winnerIdx]) {
            winnerIdx = i
        }
    }

    winner = candidates[winnerIdx]
    losers = make([]Candidate, 0, len(candidates)-1)
    for i, c := range candidates {
        if i != winnerIdx {
            losers = append(losers, c)
        }
    }
    return winner, losers
}

// better reports whether a should be preferred over the current best b.
func better(a, b Candidate) bool {
    if a.resolution() != b.resolution() {
        return a.resolution() > b.resolution()
    }
    if a.SizeBytes != b.SizeBytes {
        return a.SizeBytes > b.SizeBytes
    }
    return a.Path < b.Path
}
```

## 6. The plan file: a dry-run contract you can read

Everything above only decides; nothing touches disk. A scan's output is a `Plan` — JSON listing every group, its winner and losers, and a SHA-256 content hash of each file recorded at scan time:

```go
// internal/plan/plan.go

// FileRecord describes one image within a group, including the
// content hash used by apply to detect drift since the scan ran.
type FileRecord struct {
    Path        string  `json:"path"`
    ContentHash string  `json:"content_hash"`
    Width       int     `json:"width"`
    Height      int     `json:"height"`
    Sharpness   float64 `json:"sharpness"`
    SizeBytes   int64   `json:"size_bytes"`
}

// Group is one time-clustered, similarity-filtered set of images:
// a chosen winner and the losers to be quarantined.
type Group struct {
    ID     int          `json:"id"`
    Winner FileRecord   `json:"winner"`
    Losers []FileRecord `json:"losers"`
}

// Plan is the full output of a scan.
type Plan struct {
    Version     int       `json:"version"`
    Root        string    `json:"root"`
    GapSeconds  int       `json:"gap_seconds"`
    GeneratedAt time.Time `json:"generated_at"`
    Groups      []Group   `json:"groups"`
}
```

A plain, readable JSON file — not a database or opaque binary — is a deliberate trust-building choice: read exactly what the tool intends before running `apply`. The content hash makes `apply` safe to run later — it re-hashes every file against the plan first, and skips anything that's drifted instead of blindly moving it.

## 7. Apply and restore: move, never delete

`apply` is the only package allowed to touch the filesystem: it moves winners into `dedupe-kept/` and losers into `dedupe-quarantine/`, preserving each file's relative path. `restore` reverses it with the same plan file — nothing is ever hard-deleted. Splitting decide from act into two commands over a durable file is what makes the tool safe to experiment with: re-tune `-similarity`/`-blur` and re-run `scan` as many times as you like, and only `apply` once you're confident.

## 8. Orchestration: wiring the stages together, concurrently

`scan.Run` is the glue: resolve every file's timestamp and metrics, then feed the results through `cluster` → `simgroup` → `pick`. Decoding, hashing, and scoring is embarrassingly parallel — every file is independent — so it runs across a worker pool sized to `runtime.NumCPU()` by default:

```go
// internal/scan/scan.go

func resolveEntries(paths []string, concurrency int, progress func(index, total int, path string)) ([]entry, []Warning) {
    if concurrency <= 0 {
        concurrency = runtime.NumCPU()
    }
    if concurrency > len(paths) {
        concurrency = len(paths)
    }

    var (
        mu       sync.Mutex
        entries  []entry
        warnings []Warning
        done     int
    )

    // Routed through a buffered channel to a single reporter goroutine
    // so a worker only ever pays for a channel send, never for the
    // progress write itself — see below.
    type update struct {
        done int
        path string
    }
    progressCh := make(chan update, len(paths))
    reporterDone := make(chan struct{})
    go func() {
        defer close(reporterDone)
        for u := range progressCh {
            progress(u.done, len(paths), u.path)
        }
    }()

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
                d := done
                mu.Unlock()

                progressCh <- update{d, path}
            }
        }()
    }
    for _, path := range paths {
        jobs <- path
    }
    close(jobs)
    wg.Wait()
    close(progressCh)
    <-reporterDone

    return entries, warnings
}
```

Clustering happens only after every file resolves, so the result is identical regardless of worker order — verified by a race-detector test comparing sequential and concurrent runs. The worker pool made scanning ~3.8× faster on a 150-photo sample; the grayscale fast path above cut per-image cost by another ~10×.

The reporter-goroutine wrapping `progress` is a later bug fix: every worker used to call `progress` from inside the same mutex it uses to record results, so a slow terminal or log write throttled the entire pool. Moving that write onto a single goroutine fed by a buffered channel fixed it without changing the ordering guarantee callers depend on.

Directory discovery excludes `dedupe-kept/` and `dedupe-quarantine/` from the walk, making a re-scan after apply safe and idempotent. A `-limit` flag (default 1000) caps how many images one scan processes, keeping runtime bounded on huge libraries.

## 9. A CLI, then a local browser UI, on the same core

`cmd/dedupe/main.go` stays thin — just flag parsing per subcommand. `serve` wraps the exact same `scan.Run`/`apply.Apply`/`apply.Restore` calls as a loopback-only `net/http` handler: one in-memory `Plan` behind a mutex, HEIC re-encoded to JPEG on the fly, and only paths that are part of the loaded plan are ever served — the same trust boundary as the CLI, just enforced over HTTP.

## 10. Why the pipeline order matters

Looking back at the whole thing end to end:

```
discover → (exiftime + imagemetrics, concurrent) → cluster → simgroup → pick → plan → apply/restore
```

Each arrow is a typed package boundary, and every stage but `imagemetrics` is pure data in, pure data out — why the test suite stayed cheap to write. The lesson generalizes: when a task splits into "narrow the candidates," "confirm the match," "rank what's left," keeping those as three distinct, injectable stages pays off the first time you need to change one threshold without touching the other two.

## A real run

I pointed it at my own library — 2,023 JPEGs, defaults left mostly alone (60s gap, match 8, blur 5e6, JPEG only). The scan finished in under two minutes and came back with 126 duplicate groups, 203 photos marked for quarantine, out of 2.07 GB scanned. Nothing's been touched yet — this is still the dry-run plan, sitting there to be reviewed before a single file moves. That's the whole point: the tool tells you what it would do, and reclaiming ~12% of a library is a decision worth reading before you make it, not one worth trusting blind.

The numbers:

- 2,023 images processed
- 1m43.8s · 19.5 images/sec
- 126 duplicate groups found
- 203 photos flagged for quarantine
- 2.07 GB scanned
- ~255.3 MB reclaimable (12.3% of the library)

**Source: [github.com/a7madM/photo-dedupe](https://github.com/a7madM/photo-dedupe)**
