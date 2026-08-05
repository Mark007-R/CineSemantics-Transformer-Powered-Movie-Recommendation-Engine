"""Render assets/architecture.png.

Pillow, drawn at 2x and downsampled. Dark card with light text so it reads on
both the GitHub light and dark themes.

Run:  python assets/make_architecture.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 2
W, H = 960 * S, 600 * S
OUT = Path(__file__).with_name("architecture.png")

BG, FG, MUTED, LINE = (13, 17, 23), (201, 209, 217), (139, 148, 158), (110, 118, 129)
ACCENT, GREEN = (188, 140, 255), (63, 185, 80)
FONTS = r"C:\Windows\Fonts"


def font(n, s):
    return ImageFont.truetype(f"{FONTS}\\{n}", s * S)


f_title, f_head = font("seguisb.ttf", 15), font("seguisb.ttf", 12)
f_small, f_lbl = font("segoeui.ttf", 10), font("segoeuii.ttf", 9)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def box(x, y, w, h, c=LINE, width=2):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S],
                        radius=6 * S, outline=c, width=int(width * S))


def text(x, y, s, f=f_small, fill=MUTED, anchor="mm"):
    d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)


def _head(p0, p1, c, size=6):
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** .5, 1e-6)
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    s = size * S
    d.polygon([(x1, y1),
               (x1 - ux * s + px * s * .5, y1 - uy * s + py * s * .5),
               (x1 - ux * s - px * s * .5, y1 - uy * s - py * s * .5)], fill=c)


def arrow(pts, c=LINE, w=1.5):
    pts = [(x * S, y * S) for x, y in pts]
    for i in range(len(pts) - 1):
        d.line([pts[i], pts[i + 1]], fill=c, width=int(w * S))
    _head(pts[-2], pts[-1], c)


text(480, 26, "Semantic-Movie-Recommender — retrieval, collaborative filtering, and the eval that ranks them",
     f_title, FG)

# ── two inputs ──────────────────────────────────────────────────────────────
box(30, 54, 290, 66)
text(175, 76, "TMDB catalog", f_head, FG)
text(175, 95, "9,826 films + ~9,509 posters")
text(175, 111, "title · overview · genres")

box(640, 54, 290, 66)
text(785, 76, "MovieLens ml-latest-small", f_head, FG)
text(785, 95, "public ratings, aligned by (title, year)")
text(785, 111, "per-user temporal hold-out")

# ── content path ────────────────────────────────────────────────────────────
arrow([(175, 120), (175, 150)])
box(30, 152, 290, 66)
text(175, 174, "Embeddings — e5-base-v2", f_head, FG)
text(175, 193, "768-dim · champion of a 5-model bake-off")
text(175, 209, "poster fusion for multimodal similarity")

arrow([(175, 218), (175, 248)])
box(30, 250, 290, 60)
text(175, 271, "faiss / Milvus HNSW", f_head, FG)
text(175, 290, "~1.1 ms per query")

# ── CF path ─────────────────────────────────────────────────────────────────
arrow([(785, 120), (785, 150)])
box(640, 152, 290, 66, ACCENT)
text(785, 174, "Collaborative filtering", f_head, FG)
text(785, 193, "ItemKNN champion · ALS (Optuna-tuned)")
text(785, 209, "the only rung that moves NDCG — 5.4×")

arrow([(785, 218), (785, 248)])
box(640, 250, 290, 60)
text(785, 271, "Sequential rankers", f_head, FG)
text(785, 290, "SASRec / BERT4Rec — tie, don't beat")

# ── fusion ──────────────────────────────────────────────────────────────────
arrow([(320, 280), (376, 280)])
arrow([(640, 280), (584, 280)])
box(378, 250, 204, 60, ACCENT)
text(480, 271, "Rerank + fusion", f_head, FG)
text(480, 290, "content + CF scores blended")

# ── serving ─────────────────────────────────────────────────────────────────
arrow([(480, 310), (480, 340)])
box(180, 342, 600, 66)
text(480, 364, "FastAPI  /search  /similar  /recommend", f_head, FG)
text(480, 383, "offline mode uses cached vectors — no Milvus required")
text(480, 399, "Streamlit UI · Redis cache · Docker compose")

# ── eval harness ────────────────────────────────────────────────────────────
arrow([(480, 408), (480, 438)])
box(120, 440, 720, 82, GREEN)
text(480, 462, "Offline evaluation harness — built first, before any tuning", f_head, FG)
text(480, 481, "held-out relevance · NDCG@10 · recall@20 · MAP@20 · catalog coverage · diversity")
text(480, 499, "candidate universe is training-only items, so no test interaction leaks into training")

text(30, 548, "Every number in the README comes from this harness · cf_manifest.json records the split",
     f_lbl, GREEN, anchor="lm")
text(30, 568, "A frontier LLM matches the specialised ranker on NDCG but is off-catalog on 3.9% of picks",
     f_lbl, MUTED, anchor="lm")

img.resize((W // S, H // S), Image.LANCZOS).save(OUT, "PNG", optimize=True)
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
