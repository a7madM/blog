# magdi.work

Source for my personal site and blog, built with [Hugo](https://gohugo.io) (theme: [PaperMod](https://github.com/adityatelange/hugo-PaperMod)), hosted on [Netlify](https://www.netlify.com), and edited through [Decap CMS](https://decapcms.org) at `/admin`.

## Local development

```sh
hugo server -D
```

Open http://localhost:1313. The `-D` flag also renders draft posts.

## Writing a post

Either:

- Add a markdown file under `content/posts/` with front matter (`title`, `date`, `draft`, `tags`), or
- Go to `/admin` on the live site, log in, and write there — it commits straight to this repo.

Posts with `draft: true` are never built into the production site.

## Structure

- `content/posts/` — blog posts
- `content/about.md` — the about page
- `static/images/` — post cover images and other static assets
- `static/admin/` — Decap CMS config and admin panel
- `themes/PaperMod/` — theme, added as a git submodule
- `netlify.toml` — build config for Netlify

## Deployment

Netlify builds and deploys on every push to `main`. The custom domain and CMS login (Netlify Identity + Git Gateway) are configured in the Netlify dashboard, not in this repo.
