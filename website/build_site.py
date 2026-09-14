"""Render website tutorials from repository Markdown without duplicating content."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

import markdown
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
REPOSITORY_URL = "https://github.com/jwliaomath/CoCoFold2"
PAGES = [
    ("docs/installation.md", "installation.html", "Installation", "01 / SET UP"),
    ("examples/7zdt_7zd5/README.md", "minimal-case.html", "Minimal example", "02 / TRY A SMALL CASE"),
    ("examples/6zbh_parallel/README.md", "parallel.html", "Parallel refinement", "03 / REFINE ON TWO GPUS"),
    ("docs/component_parallel_tutorial.md", "component-preparation.html", "Component preparation", "REFERENCE / CACHES & GROUPS"),
]


def navigation(current: str) -> str:
    return "".join(
        f'<a href="{filename}"' + (' aria-current="page"' if filename == current else '')
        + f'><span>{index:02d}</span>{html.escape(label)}</a>'
        for index, (_, filename, label, _) in enumerate(PAGES, 1)
    )


def render(source_root: Path, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    for asset in (HERE / "static").iterdir():
        if not asset.is_file() or asset.is_symlink():
            raise ValueError(f"Unexpected static asset: {asset.name}")
        shutil.copyfile(asset, output_root / asset.name)
    template = (HERE / "templates" / "document.html").read_text(encoding="utf-8")
    destination = output_root / "docs"
    destination.mkdir(parents=True, exist_ok=True)
    mapping = {source: filename for source, filename, _, _ in PAGES}
    reports = []
    for index, (source, filename, label, eyebrow) in enumerate(PAGES):
        raw = (source_root / source).read_text(encoding="utf-8")
        processor = markdown.Markdown(extensions=["fenced_code", "tables", "toc", "sane_lists"])
        soup = BeautifulSoup(processor.convert(raw), "html.parser")
        title = soup.find("h1")
        if title is None:
            raise ValueError(f"Missing page title: {source}")
        title["class"] = "doc-title"
        toc_items = []
        for heading in soup.find_all(["h2", "h3"]):
            toc_items.append(f'<a class="toc-{heading.name}" href="#{heading["id"]}">{html.escape(heading.get_text())}</a>')
        rewritten = []
        for tag in soup.find_all(["a", "img"]):
            attribute = "href" if tag.name == "a" else "src"
            original = tag.get(attribute)
            if not original:
                continue
            parts = urlsplit(original)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parts.path)))
            if resolved.startswith("../") or not (source_root / resolved).exists():
                raise ValueError(f"Unresolved source link in {source}: {original}")
            if resolved in mapping:
                tag[attribute] = urlunsplit(("", "", mapping[resolved], parts.query, parts.fragment))
            else:
                kind = "tree" if (source_root / resolved).is_dir() else "blob"
                tag[attribute] = f"{REPOSITORY_URL}/{kind}/main/{resolved}"
                if parts.fragment:
                    tag[attribute] += "#" + parts.fragment
            rewritten.append({"from": original, "to": tag[attribute]})
        # Keep wide code and tables scrollable inside the article, not the page.
        for table in soup.find_all("table"):
            container = soup.new_tag("div", attrs={"class": "table-scroll", "tabindex": "0", "role": "region", "aria-label": "Scrollable reference table"})
            table.wrap(container)
        for pre in soup.find_all("pre"):
            pre["tabindex"] = "0"
            container = soup.new_tag("div", attrs={"class": "code-example"})
            pre.wrap(container)
        original_code = re.findall(r"(?ms)^```[^\n]*\n(.*?)^```[ \t]*$", raw)
        rendered_code = [pre.get_text() for pre in soup.find_all("pre")]
        if original_code != rendered_code:
            raise ValueError(f"Code block text changed during rendering: {source}")
        adjacent = []
        for other, direction in [(index - 1, "Previous"), (index + 1, "Next")]:
            if 0 <= other < len(PAGES):
                _, target, text, _ = PAGES[other]
                adjacent.append(f'<a href="{target}"><span>{direction}</span>{html.escape(text)} <span aria-hidden="true">→</span></a>')
        values = {"label": html.escape(label), "eyebrow": eyebrow,
                  "source_url": f"{REPOSITORY_URL}/blob/main/{source}",
                  "navigation": navigation(filename), "toc": "".join(toc_items),
                  "content": str(soup), "adjacent": "".join(adjacent)}
        page = template
        for key, value in values.items():
            page = page.replace("{{" + key + "}}", value)
        (destination / filename).write_text(page, encoding="utf-8", newline="\n")
        reports.append({"source": source, "output": f"docs/{filename}",
                        "source_sha256": hashlib.sha256((source_root / source).read_bytes()).hexdigest(),
                        "code_blocks": len(soup.find_all("pre")), "code_text_unchanged": True,
                        "rewritten_links": rewritten})
    return {"pages": reports, "source_policy": "Repository Markdown is authoritative; generated HTML must not be edited directly."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=HERE.parent, help="CoCoFold2 repository root containing docs/ and examples/.")
    parser.add_argument("--output", type=Path, default=HERE / "_site", help="Generated website directory; publish only this directory.")
    parser.add_argument("--report", type=Path, default=HERE / "build-report.json", help="Build provenance JSON, outside the public site.")
    args = parser.parse_args()
    report = render(args.source_root.resolve(), args.output.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"generated_pages": len(report["pages"]), "code_blocks": sum(page["code_blocks"] for page in report["pages"])}))


if __name__ == "__main__":
    main()
