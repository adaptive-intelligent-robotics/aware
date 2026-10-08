# AWARE project website

This branch contains the standalone static AWARE project page. It has independent
history from the research code on `main`. Edit and push website changes on
`aware-site`; do not merge this branch into `main`.

Expected URL: <https://adaptive-intelligent-robotics.github.io/aware/>.

## Files

- `index.html`: page metadata, hero, article, section anchors, and citation.
- `styles.css`: page styles and local font declarations.
- `script.js`: viewport video playback, citation copying, and section progress.
- `assets/`: one WebP image, seven MP4 videos, three local Latin font subsets,
  and the fonts' SIL Open Font License notices.
- `.nojekyll`: retained for compatibility with direct branch publishing.
- `.github/workflows/pages.yml`: deployment on pushes to `aware-site`.

No framework, package installation, build step, CDN, analytics, or external
font requests are required. The site is approximately 67 MB, predominantly
video. Runtime asset URLs are relative and work below the `/aware/` path.

## Work on the website

From a fresh research checkout on `main`, create a separate worktree:

```bash
git fetch origin aware-site:refs/remotes/origin/aware-site
git worktree add -b aware-site .worktrees/aware-site origin/aware-site
cd .worktrees/aware-site
```

If the branch already exists locally, use
`git worktree add .worktrees/aware-site aware-site` instead. If the worktree
already exists, open it directly. This keeps the research checkout on `main`.

Alternatively, clone only the website branch:

```bash
git clone --single-branch --branch aware-site https://github.com/adaptive-intelligent-robotics/aware.git aware-site
cd aware-site
```

## Preview locally

From the research checkout root:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 \
  --directory .worktrees/aware-site
```

Visit <http://localhost:8080/>. Press Ctrl+C to stop the server.
From a standalone website clone or inside the website worktree, omit
`--directory .worktrees/aware-site`.
The ignored local backup in the research checkout can also be previewed with
`python3 -m http.server 8080 --bind 127.0.0.1 --directory site`.

The AWARE project video before the Abstract uses native play/pause and volume
controls, starts with sound enabled, and plays only when requested without
looping. The six muted, looping demo clips play automatically only while visible,
unless reduced motion is enabled or the visitor has paused them. The article, section
links, and media controls work without JavaScript. Both Copy BibTeX buttons stay
visible. With JavaScript, they use the Clipboard API on HTTPS or localhost and
try a legacy copy fallback when that API is unavailable. If copying is blocked,
the citation is selected for manual copying. The text is always available to select.

## Enable GitHub Pages

After pushing `aware-site`, configure the repository once:

1. In **Settings → Pages**, select **GitHub Actions** as the source.
2. In **Settings → Environments → github-pages**, allow the `aware-site` branch
   to deploy. If restricting deployment branches, select **Selected branches
   and tags** and add a **Branch** rule for `aware-site`.
3. Push a website update, or rerun its deployment workflow in **Actions** if
   the first run happened before Pages was enabled.
4. Open the URL shown by the successful deployment.

On GitHub Free for organisations, the repository must be public before enabling
Pages. Private-repository Pages requires GitHub Team or Enterprise. Publishing
does not change repository visibility; the published site is public unless
separate access controls are configured.

See GitHub's [publishing-source documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Deploy updates

Commit and push from the website worktree:

```bash
git add index.html styles.css script.js assets README.md .nojekyll .github/workflows/pages.yml
git commit -m "Update AWARE project website"
git push origin aware-site
```

The workflow stages only `index.html`, `styles.css`, `script.js`, `.nojekyll`,
and `assets/`, then uploads and deploys them using GitHub's Pages actions.
The README and workflow files are excluded from the public artifact. Deployments
are serialised, and in-progress deployments are allowed to finish.

The workflow runs on pushes to `aware-site`. To retry a failed deployment, use
**Actions → Deploy AWARE website → Re-run jobs**. There is no manual dispatch
button: GitHub requires a manually dispatched workflow to also exist on the
default branch, and this workflow is maintained entirely on `aware-site`.

## Update the snapshot

### Project resource links

The resource row below the hero tags is in `nav.project-links` in `index.html`.
The Code link points to the research repository. arXiv, Paper (PDF), and
Supplementary (PDF) are disabled placeholders labelled **Coming soon**.

To activate a resource, replace its `data-href` attribute with `href`, enter the
actual destination, remove `role="link"` and `aria-disabled="true"`, and remove
the `<span class="resource-status">Coming soon</span>` label. External links
use an absolute HTTPS URL. For local PDFs, add `aware-paper.pdf` or
`aware-supplementary.pdf` to `assets/` and use the relative paths already provided
in the placeholders. Files in `assets/` are included automatically in deployment.

### Article and media

Edit `index.html` for text, metadata, tags, or citation changes. Replace files in
`assets/` for media changes, retaining the referenced filenames or updating their
relative URLs. When adding or renaming an article heading, update its `id` and
the matching link in `nav[aria-label="Project sections"]`.

The page was copied from another repository and does not automatically track
that repository or changes to the research code. Its original article text and
statistics are preserved. The original `WM_prediction.mov` reference uses the
included `WM_prediction.mp4`; the commented-out score demo is not included.

Archivo, Newsreader, and Space Mono were bundled with their SIL Open Font License
notices. Keep those notices when redistributing the site.
