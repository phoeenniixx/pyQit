#!/usr/bin/env python3
"""Draft a changelog entry from merged pull requests on GitHub.

    python build_tools/changelog.py
    python build_tools/changelog.py --since-pr 20
    python build_tools/changelog.py --since-date 2026-06-01

Needs the GitHub CLI (``gh``), logged in (``gh auth login``). Queries merged
PRs against the repository's ``main`` branch on GitHub itself, not local git
state, so the result does not depend on which branch or commit you happen to
have checked out.

A PR's ``bug`` / ``enhancement`` / ``documentation`` / ``maintenance`` label
decides its section. Many PRs in this repo carry no label, so an unlabeled PR
falls back to a keyword match on its title. A PR that matches no label and no
keyword lands in "Needs a label" rather than a guessed section, so labeling
it and re-running is the fix, not moving it by hand.

Nothing is written to CHANGELOG.md automatically. Paste the output in
yourself, replacing the whole "## Unreleased" section each time -- it is
meant to be regenerated in full, not appended to. Until you rename that
heading to a real version and start a fresh "## Unreleased" above it,
nothing counts as released, so a plain re-run always shows every merged PR
again. Renaming it to a version heading is what makes "--since-pr" (or the
auto-detected default) start narrowing later runs.
"""

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

SECTIONS = {
    "bug": "Bug fixes",
    "enhancement": "Enhancements",
    "documentation": "Documentation",
    "maintenance": "Maintenance",
}

# Not a real GitHub label; where a PR lands when no label and no keyword
# rule below matched it. Kept separate from SECTIONS so it renders last and
# is never treated as a labelable category to search for on a PR.
UNCLASSIFIED = "unclassified"
UNCLASSIFIED_HEADING = "Needs a label"

# Tried in order against the title, for PRs with no matching label.
KEYWORD_RULES = [
    ("bug", re.compile(r"\bfix|bug|hotfix|\bpatch", re.I)),
    (
        "documentation",
        re.compile(r"\bdocs?\b|readme|tutorial|notebook|changelog|contributing", re.I),
    ),
    (
        "maintenance",
        re.compile(
            r"\bchore|refactor|cleanup|bump|deps?\b|dependabot|\bci\b|revert"
            r"|merge branch",
            re.I,
        ),
    ),
]

CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"$ {' '.join(cmd)}\n{result.stderr.strip()}")
    return result.stdout


def _require_gh() -> None:
    if shutil.which("gh") is None:
        sys.exit("Needs the GitHub CLI ('gh'). Install it, then 'gh auth login'.")
    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        sys.exit("'gh' is not logged in. Run 'gh auth login' first.")


def _default_repo() -> str:
    """OWNER/REPO, read from the 'origin' remote rather than hardcoded."""
    url = _run(["git", "remote", "get-url", "origin"]).strip()
    match = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/]+?)(\.git)?$", url)
    if not match:
        sys.exit(f"Could not read a GitHub repo from the 'origin' remote: {url!r}")
    return match.group("repo")


def _last_released_pr() -> int | None:
    """Highest PR number under the most recent *released* heading.

    "## Unreleased" is not a release: it gets fully regenerated on every run
    until you rename it to a real version and start a fresh, empty
    "Unreleased" above it. Counting PRs already listed there as "handled"
    would mean a plain re-run stops reporting anything the moment you first
    paste a draft in, which is the wrong direction -- there is nothing to
    protect against re-showing until something has actually shipped.
    """
    if not CHANGELOG_PATH.exists():
        return None
    sections = re.split(r"^## (.+)$", CHANGELOG_PATH.read_text(), flags=re.M)
    for heading, body in zip(sections[1::2], sections[2::2]):
        if heading.strip().lower() == "unreleased":
            continue
        numbers = [int(n) for n in re.findall(r"\[#(\d+)\]", body)]
        return max(numbers) if numbers else None
    return None


def fetch_merged_prs(repo: str, base: str, limit: int) -> list[dict]:
    out = _run(
        [
            "gh",
            "pr",
            "list",
            "-R",
            repo,
            "--base",
            base,
            "--state",
            "merged",
            "--limit",
            str(limit),
            "--json",
            "number,title,author,labels,mergedAt,url",
        ]
    )
    return json.loads(out)


def classify(pr: dict) -> str:
    labels = {label["name"] for label in pr["labels"]}
    for key in SECTIONS:
        if key in labels:
            return key
    for key, pattern in KEYWORD_RULES:
        if pattern.search(pr["title"]):
            return key
    return UNCLASSIFIED


def render(prs: list[dict], heading: str) -> str:
    all_sections = {**SECTIONS, UNCLASSIFIED: UNCLASSIFIED_HEADING}
    buckets: dict[str, list[dict]] = {key: [] for key in all_sections}
    for pr in prs:
        buckets[classify(pr)].append(pr)

    lines = [
        "<!-- Draft, not a source of truth. A PR's label decides its "
        "section; an unlabeled PR is guessed from its title. Anything "
        "matching neither ends up in 'Needs a label' -- label it on "
        "GitHub and re-run rather than moving it here by hand. -->",
        "",
        f"## {heading}",
        "",
    ]
    for key, section_heading in all_sections.items():
        entries = buckets[key]
        if not entries:
            continue
        lines.append(f"### {section_heading}")
        lines.append("")
        for pr in entries:
            login = pr["author"].get("login", "unknown")
            lines.append(
                f"- {pr['title']} ([#{pr['number']}]({pr['url']})) "
                f"by [@{login}](https://github.com/{login})"
            )
        lines.append("")

    contributors = sorted(
        {pr["author"]["login"] for pr in prs if not pr["author"].get("is_bot", False)},
        key=str.lower,
    )
    if contributors:
        lines.append("### Contributors")
        lines.append("")
        lines.append("Thanks to the following people for this release:")
        lines.append("")
        lines.append(
            ", ".join(
                f"[@{login}](https://github.com/{login})" for login in contributors
            )
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", default=None, help="OWNER/REPO. Default: read from 'origin'."
    )
    parser.add_argument(
        "--base", default="main", help="Base branch merged PRs target. Default: main."
    )
    parser.add_argument(
        "--since-pr", type=int, default=None, help="Only PRs numbered above this."
    )
    parser.add_argument(
        "--since-date", default=None, help="Only PRs merged after this (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--limit", type=int, default=300, help="Max merged PRs to fetch. Default 300."
    )
    parser.add_argument(
        "--title", default="Unreleased", help="Section heading. Default: Unreleased."
    )
    parser.add_argument("--output", default=None, help="Also write the markdown here.")
    args = parser.parse_args()

    _require_gh()
    repo = args.repo or _default_repo()

    since_pr = args.since_pr
    if since_pr is None and args.since_date is None:
        since_pr = _last_released_pr()

    prs = fetch_merged_prs(repo, args.base, args.limit)

    if since_pr is not None:
        prs = [pr for pr in prs if pr["number"] > since_pr]
    if args.since_date is not None:
        prs = [pr for pr in prs if pr["mergedAt"][:10] > args.since_date]

    if not prs:
        sys.exit("No merged PRs in range. Nothing to report.")

    prs.sort(key=lambda pr: pr["number"], reverse=True)

    markdown = render(prs, args.title)
    print(markdown)

    if args.output:
        Path(args.output).write_text(markdown)


if __name__ == "__main__":
    main()
