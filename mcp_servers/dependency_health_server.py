"""
MCP server: GitHub dependency-health tools.

Exposes four tools over stdio that hit the public GitHub REST API.
Spawned by clients via:  python mcp_servers/dependency_health_server.py
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dependency-health")

GH = "https://api.github.com"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
if os.environ.get("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"


def _gh(path: str, params: dict | None = None):
    r = requests.get(f"{GH}{path}", headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def get_repo(repo: str) -> dict:
    """Fetch GitHub repo metadata: stars, forks, language, archived flag, last push, license. `repo` is 'owner/name'."""
    d = _gh(f"/repos/{repo}")
    return {
        k: d.get(k)
        for k in (
            "full_name",
            "description",
            "language",
            "stargazers_count",
            "forks_count",
            "open_issues_count",
            "pushed_at",
            "created_at",
            "archived",
            "license",
        )
    }


@mcp.tool()
def get_recent_commits(repo: str, days: int = 90) -> dict:
    """Commits in the last N days. Returns commit count, unique author count, latest commit message."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    commits = _gh(f"/repos/{repo}/commits", params={"since": since, "per_page": 100})
    authors = {
        c["commit"]["author"]["email"]
        for c in commits
        if c.get("commit", {}).get("author")
    }
    last_msg = commits[0]["commit"]["message"].splitlines()[0] if commits else ""
    return {
        "window_days": days,
        "commit_count": len(commits),
        "unique_authors": len(authors),
        "latest_message": last_msg[:140],
    }


@mcp.tool()
def get_issue_summary(repo: str) -> dict:
    """Open-issue counts, ages, and a sample of titles. Use to assess support burden."""
    issues = _gh(f"/repos/{repo}/issues", params={"state": "open", "per_page": 100})
    issues = [i for i in issues if "pull_request" not in i]
    now = datetime.now(timezone.utc)
    ages = [
        (now - datetime.fromisoformat(i["created_at"].replace("Z", "+00:00"))).days
        for i in issues
    ]
    return {
        "open_issue_count": len(issues),
        "median_age_days": int(sorted(ages)[len(ages) // 2]) if ages else 0,
        "oldest_age_days": max(ages) if ages else 0,
        "sample_titles": [i["title"] for i in issues[:5]],
    }


@mcp.tool()
def get_release_info(repo: str) -> dict:
    """Most-recent releases: tag, days since latest, count seen."""
    rels = _gh(f"/repos/{repo}/releases", params={"per_page": 10})
    if not rels:
        return {
            "release_count_total_seen": 0,
            "latest_tag": None,
            "days_since_latest": None,
        }
    latest = rels[0]
    days = (
        datetime.now(timezone.utc)
        - datetime.fromisoformat(latest["published_at"].replace("Z", "+00:00"))
    ).days
    return {
        "release_count_total_seen": len(rels),
        "latest_tag": latest["tag_name"],
        "days_since_latest": days,
        "latest_name": latest.get("name"),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
