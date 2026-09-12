"""render_schematic · overview 示意图渲染器（HTML → PNG）。

源文件: docs/img/benchmark/overview_schematic.html（自包含 HTML+CSS，
paper-figure-html 模板设计语言）。本脚本用本机 Edge/Chrome headless 以
2x 设备缩放截图为 README 用的 PNG。改 HTML 后重跑本脚本即可。

用法: python scripts/render_schematic.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "img" / "benchmark" / "overview_schematic.html"
OUT = REPO / "docs" / "img" / "benchmark" / "overview_schematic.png"
PAGE_W = 1180
PAGE_H = 560
SCALE = 2

CANDIDATE_BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser() -> str:
    for p in CANDIDATE_BROWSERS:
        if Path(p).exists():
            return p
    raise FileNotFoundError("no Edge/Chrome found in standard locations")


def main() -> None:
    browser = find_browser()
    url = HTML.as_uri()
    before = OUT.stat().st_mtime if OUT.exists() else 0
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=%d" % SCALE,
        "--window-size=%d,%d" % (PAGE_W, PAGE_H),
        "--screenshot=%s" % OUT,      # no embedded quotes: Edge would ignore them
        "--default-background-color=FFFFFFFF",
        url,
    ]
    print("[schematic] browser:", browser)
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    assert OUT.exists() and OUT.stat().st_mtime > before, \
        "screenshot not regenerated (Edge failed silently)"
    print(f"[schematic] wrote {OUT} ({OUT.stat().st_size} bytes, {PAGE_W}x{PAGE_H} @{SCALE}x)")


if __name__ == "__main__":
    main()
