"""Capture docs/demo.gif from `outrun serve --demo --port 8765` (start it first). Usage: make_demo_gif.py docs/demo.gif"""

import io
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

out = sys.argv[1]
frames = []


def snap(page, n=1, d=0.0):
    time.sleep(d)
    for _ in range(n):
        frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.goto("http://127.0.0.1:8765/")
    pg.wait_for_timeout(3000)
    pg.select_option("#scenario-select", "may20-2019")
    pg.wait_for_timeout(2500)
    snap(pg, 2)
    for s in (3, 6, 9, 11):
        pg.evaluate(
            "s=>{const e=document.getElementById('slot-slider');e.value=s;e.dispatchEvent(new Event('input',{bubbles:true}))}",
            s,
        )
        snap(pg, 1, 0.8)
    pg.check("#swipe-toggle")
    snap(pg, 2, 0.8)
    pg.click("[data-tab=route]")
    pg.wait_for_timeout(800)
    snap(pg, 1)
    pg.mouse.click(520, 380)
    pg.wait_for_timeout(2500)
    snap(pg, 3)
    pg.click("[data-tab=shelters]")
    pg.wait_for_timeout(2500)
    snap(pg, 3)
    pg.click("[data-tab=evac]")
    pg.wait_for_timeout(800)
    snap(pg, 1)
    pg.click("#evac-run")
    for _ in range(4):
        snap(pg, 1, 1.2)
    pg.click("[data-tab=results]")
    pg.wait_for_timeout(1500)
    snap(pg, 3)
    b.close()
w = 800
fr = [
    f.resize((w, int(f.height * w / f.width)), Image.LANCZOS).quantize(128, dither=Image.Dither.NONE)
    for f in frames
]
fr[0].save(out, save_all=True, append_images=fr[1:], duration=900, loop=0, optimize=True)
print(len(fr))
