# Project website

The static homepage and documentation are hosted at
https://jwliaomath.github.io/CoCoFold2/.

The homepage presents the preprint, method figures and citation. The primary
user guide is https://jwliaomath.gitbook.io/cocofold2/; existing HTML tutorial URLs
remain available and are still generated from the same repository Markdown.
The academic homepage layout is inspired by Academic Project Page Template
(https://github.com/eliahuhorwitz/Academic-project-page-template), implemented
independently with local CSS and assets. No template JavaScript, fonts, analytics
or third-party stylesheet dependencies are required.

## Build locally

Use a separate Python environment for website tools; the model environment is
not required. From the repository root:

```bash
python -m pip install -r website/requirements-site.txt
python website/build_site.py
python website/check_site.py
python -m http.server 8000 --bind 127.0.0.1 --directory website/_site
```

Open http://127.0.0.1:8000/. Only `website/_site/` is published; generation
scripts, Markdown sources and audit reports are not included in that artifact.

## Update content

- Edit the homepage and shared CSS/JavaScript in `website/static/`.
- Edit tutorials in their existing `docs/` and `examples/` Markdown files.
  `build_site.py` maps selected documents to site pages and converts their links.
- Edit the shared document layout in `website/templates/document.html`.
- Do not edit generated HTML or copy tutorial Markdown into the website folder.

The generator verifies that fenced code text is unchanged and writes source
hashes to `website/build-report.json`. `check_site.py` validates generated
page links, fragments and local assets, including the `/CoCoFold2/` URL prefix.
It does not run the displayed training commands or check external services.

## Deployment

The Website workflow builds and checks pull requests. Only a successful build
on `main` can deploy to the `github-pages` environment. Site dependencies are
separate from the CPU/GPU dependencies. The workflow rebuilds on each main
update, so edits to the original Markdown are reflected in the site.

The homepage schematic is illustrative; it does not contain experimental data.

## Paper figures

`static/paper-figure1.webp` and `static/paper-figure3.webp` are losslessly encoded
web renderings of Figures 1 and 3 from the authors' preprint (https://doi.org/10.65215/LTSpreprints.2026.09.15.000338).
Their content has not been redrawn or altered. These figure assets are not
covered by the software Apache-2.0 license; see the source publication for
figure reuse terms.
