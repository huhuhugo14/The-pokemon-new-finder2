"""
PSA 10 Profit Scanner — Pokémon Card Grading & ROI Tool
Streamlit + OpenCV + eBay Browse API
"""

import streamlit as st
import cv2
import numpy as np
import requests
import yaml
import os
import io
import json
import time
import base64
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG_PATH = "config.yaml"

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}

cfg = load_config()

EBAY_APP_ID   = cfg.get("ebay", {}).get("app_id",   os.getenv("EBAY_APP_ID", ""))
EBAY_CERT_ID  = cfg.get("ebay", {}).get("cert_id",  os.getenv("EBAY_CERT_ID", ""))
EBAY_DEV_ID   = cfg.get("ebay", {}).get("dev_id",   os.getenv("EBAY_DEV_ID", ""))

GRADING_FEE   = 25.0
SHIPPING_EST  = 5.0
MIN_IMAGE_PX  = 1000
CENTER_THRESH = 0.55   # 55/45 rule — reject if ratio > 0.55
GLINT_THRESH  = 245    # brightness threshold for glare
SCRATCH_THRESH= 0.003  # fraction of pixels flagged by Laplacian
WHITE_THRESH  = 220    # pixel value considered "white"
WHITE_CLUSTER = 30     # min white pixels in corner patch to flag whitening

# High-ROI target sets (Gap Finder seed list)
HIGH_ROI_SETS = [
    {"set": "Base Set (1st Edition)", "cards": ["Charizard", "Blastoise", "Venusaur", "Raichu", "Nidoking"], "note": "Legendary — PSA 10 spreads often $5k–$500k+"},
    {"set": "Base Set Shadowless", "cards": ["Charizard", "Blastoise", "Venusaur", "Ninetales"], "note": "Shadowless holos extremely scarce in gem mint"},
    {"set": "Base Set Unlimited", "cards": ["Charizard", "Blastoise", "Venusaur", "Machamp"], "note": "High volume, easier to find raw — PSA 10 still $500–$5k+"},
    {"set": "Jungle", "cards": ["Clefable", "Scyther", "Snorlax", "Vaporeon"], "note": "Notorious for centering issues → PSA 10 premium huge"},
    {"set": "Fossil", "cards": ["Gengar", "Lapras", "Moltres", "Zapdos"], "note": "Thick cards prone to dings — gem copies command big premium"},
    {"set": "Team Rocket", "cards": ["Dark Charizard", "Dark Blastoise", "Here Comes Team Rocket!"], "note": "Dark cards show scratches easily → PSA 10 scarcity"},
    {"set": "Neo Genesis", "cards": ["Lugia", "Typhlosion", "Feraligatr", "Meganium"], "note": "Lugia PSA 10 routinely $10k+ — raw copies under $200 exist"},
    {"set": "Neo Revelation", "cards": ["Ho-Oh", "Entei", "Raikou", "Suicune"], "note": "Legendary holos — massive PSA 10 upside"},
    {"set": "Neo Destiny", "cards": ["Shining Charizard", "Shining Magikarp", "Dark Espeon"], "note": "Shining series = among rarest PSA 10s in hobby"},
    {"set": "Aquapolis / Skyridge", "cards": ["Charizard", "Articuno", "Celebi", "Jolteon"], "note": "e-Reader era — extremely print-line prone. PSA 10 rarities"},
    {"set": "Expedition", "cards": ["Charizard", "Blastoise", "Venusaur"], "note": "Old holofoil tech = lots of holo damage → huge gem premium"},
]

# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────
@dataclass
class GradingResult:
    centering_front_lr:  float = 0.0
    centering_front_tb:  float = 0.0
    centering_back_lr:   float = 0.0
    centering_back_tb:   float = 0.0
    centering_flag:      bool  = False
    corners_flag:        bool  = False
    corner_labels:       list  = field(default_factory=list)
    glint_flag:          bool  = False
    scratch_flag:        bool  = False
    resolution_flag:     bool  = False
    resolution_px:       tuple = (0, 0)
    overall_pass:        bool  = False
    psa10_prob:          float = 0.0
    psa9_prob:           float = 0.0
    psa8_prob:           float = 0.0
    psa7_below_prob:     float = 0.0
    reasoning:           list  = field(default_factory=list)
    corner_patches:      list  = field(default_factory=list)  # numpy arrays

@dataclass
class ListingResult:
    title:        str   = ""
    price:        float = 0.0
    url:          str   = ""
    image_urls:   list  = field(default_factory=list)
    grading:      Optional[GradingResult] = None
    psa10_value:  float = 0.0
    net_profit:   float = 0.0
    roi_pct:      float = 0.0
    verdict:      str   = "PENDING"

# ─────────────────────────────────────────────
# CV GRADING ENGINE
# ─────────────────────────────────────────────

def fetch_image_from_url(url: str) -> Optional[np.ndarray]:
    """Download image and convert to OpenCV BGR array."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def check_resolution(img: np.ndarray) -> tuple[bool, tuple]:
    """Reject if either dimension < MIN_IMAGE_PX."""
    h, w = img.shape[:2]
    passed = (w >= MIN_IMAGE_PX and h >= MIN_IMAGE_PX)
    return passed, (w, h)

def detect_card_borders(img: np.ndarray) -> Optional[tuple]:
    """
    Use Canny edge detection to find the card rectangle.
    Returns (left, right, top, bottom) pixel positions or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Find contours and pick the largest quadrilateral
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (img.shape[0] * img.shape[1] * 0.10):  # must be >10% of image
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area

    if best is None:
        # Fall back: bounding rect of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x, x + w, y, y + h)

    pts = best.reshape(4, 2)
    left   = int(pts[:, 0].min())
    right  = int(pts[:, 0].max())
    top    = int(pts[:, 1].min())
    bottom = int(pts[:, 1].max())
    return (left, right, top, bottom)

def check_centering(img: np.ndarray) -> tuple[float, float, bool, list]:
    """
    Calculate L/R and T/B centering ratios.
    Returns (lr_ratio, tb_ratio, flag, reasons).
    Ratio = larger_border / (larger_border + smaller_border).
    PSA 10 requires <= 55/45 on front.
    """
    h, w = img.shape[:2]
    borders = detect_card_borders(img)
    reasons = []

    if borders is None:
        return 0.5, 0.5, False, ["Could not detect card borders for centering"]

    left, right, top, bottom = borders

    border_left   = left
    border_right  = w - right
    border_top    = top
    border_bottom = h - bottom

    # Avoid divide-by-zero
    lr_total = border_left + border_right
    tb_total = border_top  + border_bottom

    if lr_total == 0 or tb_total == 0:
        return 0.5, 0.5, False, ["Border width too small to measure centering"]

    lr_ratio = max(border_left, border_right) / lr_total
    tb_ratio = max(border_top, border_bottom) / tb_total

    flag = False
    if lr_ratio > CENTER_THRESH:
        flag = True
        reasons.append(f"L/R centering {lr_ratio:.1%} — exceeds 55/45 (PSA 10 fail)")
    if tb_ratio > CENTER_THRESH:
        flag = True
        reasons.append(f"T/B centering {tb_ratio:.1%} — exceeds 55/45 (PSA 10 fail)")

    return lr_ratio, tb_ratio, flag, reasons

def extract_corner_patches(img: np.ndarray, patch_pct: float = 0.08) -> list:
    """Extract 4 corner patches for whitening analysis."""
    h, w = img.shape[:2]
    ph = int(h * patch_pct)
    pw = int(w * patch_pct)
    ph = max(ph, 20)
    pw = max(pw, 20)

    corners = [
        ("TL", img[0:ph, 0:pw]),
        ("TR", img[0:ph, w-pw:w]),
        ("BL", img[h-ph:h, 0:pw]),
        ("BR", img[h-ph:h, w-pw:w]),
    ]
    return corners

def check_corners_whitening(img: np.ndarray) -> tuple[bool, list, list]:
    """
    Scan corners for white-pixel clusters on blue-back images.
    Returns (flag, corner_labels, patches_bgr).
    """
    corners = extract_corner_patches(img)
    flagged = []
    patches = []

    for label, patch in corners:
        if patch.size == 0:
            patches.append(patch)
            continue

        # Check if this looks like a blue-back (dominant blue hue)
        b, g, r = cv2.split(patch)
        is_blue_dominant = (b.mean() > g.mean() + 15) and (b.mean() > r.mean() + 15)

        # Count white pixels (all channels > WHITE_THRESH)
        white_mask = (r > WHITE_THRESH) & (g > WHITE_THRESH) & (b > WHITE_THRESH)
        white_count = int(white_mask.sum())

        if white_count >= WHITE_CLUSTER:
            flagged.append(f"{label} corner: {white_count} white pixels detected (whitening)")

        # Highlight white pixels for display
        display = patch.copy()
        display[white_mask] = [0, 0, 255]  # red overlay on white pixels
        patches.append(display)

    flag = len(flagged) > 0
    return flag, flagged, [p for _, p in corners]

def check_glint_and_scratches(img: np.ndarray) -> tuple[bool, bool, list]:
    """
    1. Detect glint (high-brightness glare in holo area).
    2. If glint found, apply Laplacian to look for scratches/print lines.
    Returns (glint_flag, scratch_flag, reasons).
    """
    reasons = []
    h, w = img.shape[:2]

    # Holo area = center 60% of card
    y1 = int(h * 0.20)
    y2 = int(h * 0.70)
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    holo_region = img[y1:y2, x1:x2]

    if holo_region.size == 0:
        return False, False, ["Could not isolate holo region"]

    gray_holo = cv2.cvtColor(holo_region, cv2.COLOR_BGR2GRAY)

    # Glint: fraction of pixels above GLINT_THRESH
    glint_mask = gray_holo > GLINT_THRESH
    glint_fraction = glint_mask.sum() / gray_holo.size

    glint_flag = glint_fraction > 0.02   # >2% of holo area is blown out
    scratch_flag = False

    if glint_flag:
        reasons.append(f"Glare/glint detected ({glint_fraction:.1%} of holo area overexposed)")

        # Laplacian scratch check within the glint zones
        lap = cv2.Laplacian(gray_holo, cv2.CV_64F)
        lap_abs = np.abs(lap)

        # Within glint area, look for high-variance edges (scratches appear as dark lines)
        glint_region = lap_abs[glint_mask]
        if glint_region.size > 0:
            high_var_fraction = (glint_region > 40).sum() / glint_region.size
            if high_var_fraction > SCRATCH_THRESH:
                scratch_flag = True
                reasons.append(f"Potential scratch/print lines in glare zone ({high_var_fraction:.2%} edge density)")

    return glint_flag, scratch_flag, reasons

def compute_grade_probabilities(gr: GradingResult) -> GradingResult:
    """
    Strict Bayesian-style probability distribution based on flags.
    Base state: pristine card with no flags.
    Each flag reduces PSA 10 probability severely.
    """
    p10 = 0.82   # base: a truly mint raw card still only ~82% to gem

    # Resolution fail → can't assess, probability collapses
    if gr.resolution_flag:
        gr.psa10_prob     = 0.0
        gr.psa9_prob      = 0.0
        gr.psa8_prob      = 0.0
        gr.psa7_below_prob= 100.0
        return gr

    if gr.centering_flag:
        p10 -= 0.60   # centering is the #1 PSA 10 killer
    if gr.corners_flag:
        p10 -= 0.50   # corner whitening almost always disqualifies PSA 10
    if gr.glint_flag:
        p10 -= 0.25   # glare may just be photography; penalize but less harshly
    if gr.scratch_flag:
        p10 -= 0.40   # scratches detected → major deduction

    p10 = max(p10, 0.0)

    # Distribute remaining probability
    total_flags = sum([gr.centering_flag, gr.corners_flag, gr.scratch_flag])
    if total_flags == 0:
        p9  = 0.12
        p8  = 0.04
        p7b = 0.02
    elif total_flags == 1:
        p9  = 0.35
        p8  = 0.20
        p7b = 0.15
    else:
        p9  = 0.20
        p8  = 0.25
        p7b = 0.30

    # Normalize
    total = p10 + p9 + p8 + p7b
    if total > 0:
        p10 /= total
        p9  /= total
        p8  /= total
        p7b /= total

    gr.psa10_prob      = round(p10 * 100, 1)
    gr.psa9_prob       = round(p9  * 100, 1)
    gr.psa8_prob       = round(p8  * 100, 1)
    gr.psa7_below_prob = round(p7b * 100, 1)
    return gr

def run_full_grading(image_urls: list) -> GradingResult:
    """
    Run all CV checks across all images in a listing.
    Strictest flag from any image is used.
    """
    gr = GradingResult()

    if not image_urls:
        gr.reasoning.append("No images found in listing")
        gr.resolution_flag = True
        gr.overall_pass = False
        return compute_grade_probabilities(gr)

    images = []
    for url in image_urls[:6]:   # cap at 6 images per listing
        img = fetch_image_from_url(url)
        if img is not None:
            images.append(img)
        time.sleep(0.1)

    if not images:
        gr.reasoning.append("All image downloads failed")
        gr.resolution_flag = True
        gr.overall_pass = False
        return compute_grade_probabilities(gr)

    # ── Resolution check (first image)
    res_pass, res_px = check_resolution(images[0])
    gr.resolution_px = res_px
    if not res_pass:
        gr.resolution_flag = True
        gr.reasoning.append(f"REJECTED: Stock photo / low resolution ({res_px[0]}×{res_px[1]}px < {MIN_IMAGE_PX}px required)")
        gr.overall_pass = False
        return compute_grade_probabilities(gr)

    # ── Run checks on each image, accumulate worst-case flags
    for idx, img in enumerate(images):
        img_label = f"Image {idx+1}"

        # Centering
        lr, tb, c_flag, c_reasons = check_centering(img)
        if idx == 0:  # front of card
            gr.centering_front_lr = lr
            gr.centering_front_tb = tb
        elif idx == 1:
            gr.centering_back_lr = lr
            gr.centering_back_tb = tb
        if c_flag:
            gr.centering_flag = True
            gr.reasoning.extend([f"{img_label}: {r}" for r in c_reasons])

        # Corners (check every image for whitening)
        cor_flag, cor_labels, patches = check_corners_whitening(img)
        if cor_flag:
            gr.corners_flag = True
            gr.reasoning.extend([f"{img_label}: {r}" for r in cor_labels])
        if patches and not gr.corner_patches:
            gr.corner_patches = patches   # keep first set for display

        # Glint / scratch
        g_flag, s_flag, g_reasons = check_glint_and_scratches(img)
        if g_flag:
            gr.glint_flag = True
        if s_flag:
            gr.scratch_flag = True
        if g_reasons:
            gr.reasoning.extend([f"{img_label}: {r}" for r in g_reasons])

    # ── Determine overall pass (STRICT mode)
    hard_fails = [gr.centering_flag, gr.corners_flag, gr.scratch_flag, gr.resolution_flag]
    gr.overall_pass = not any(hard_fails)   # glint alone = warning, not reject
    if not gr.overall_pass and not gr.reasoning:
        gr.reasoning.append("One or more hard-fail conditions detected")
    if gr.overall_pass:
        gr.reasoning.append("✓ All automated checks passed — recommend manual review before purchase")

    gr = compute_grade_probabilities(gr)
    return gr

# ─────────────────────────────────────────────
# eBay BROWSE API
# ─────────────────────────────────────────────

def get_ebay_oauth_token() -> Optional[str]:
    """Get eBay OAuth application token via Client Credentials."""
    if not EBAY_APP_ID or not EBAY_CERT_ID:
        return None
    credentials = base64.b64encode(f"{EBAY_APP_ID}:{EBAY_CERT_ID}".encode()).decode()
    url = "https://api.ebay.com/identity/v1/oauth2/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {credentials}",
    }
    data = {"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"}
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=10)
        resp.raise_for_status()
        return resp.json().get("access_token")
    except Exception as e:
        st.error(f"eBay OAuth failed: {e}")
        return None

def search_ebay_listings(query: str, min_price: float = 1.0, max_price: float = 100.0,
                          limit: int = 20) -> list:
    """Search eBay using Browse API for raw Pokémon cards."""
    token = get_ebay_oauth_token()
    if not token:
        return _mock_listings(query)   # Demo mode if no token

    url = "https://api.ebay.com/buy/browse/v1/item_summary/search"
    params = {
        "q": f"{query} pokemon card raw ungraded",
        "category_ids": "183454",   # TCG Individual Cards
        "filter": f"price:[{min_price}..{max_price}],priceCurrency:USD,conditions:{{USED|FOR_PARTS_OR_NOT_WORKING|UNSPECIFIED}}",
        "sort": "price",
        "limit": limit,
        "fieldgroups": "EXTENDED",
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("itemSummaries", [])
        return _parse_ebay_items(items)
    except Exception as e:
        st.warning(f"eBay API error: {e} — using demo data")
        return _mock_listings(query)

def _parse_ebay_items(items: list) -> list:
    listings = []
    for item in items:
        try:
            price = float(item.get("price", {}).get("value", 0))
            imgs = [img.get("imageUrl", "") for img in item.get("additionalImages", [])]
            thumb = item.get("image", {}).get("imageUrl", "")
            if thumb:
                imgs.insert(0, thumb)
            listings.append({
                "title":      item.get("title", "Unknown"),
                "price":      price,
                "url":        item.get("itemWebUrl", "#"),
                "image_urls": imgs,
            })
        except Exception:
            continue
    return listings

def _mock_listings(query: str) -> list:
    """Demo listings when no eBay credentials are configured."""
    return [
        {
            "title": f"[DEMO] {query} Holo Rare — Base Set — Ungraded Raw",
            "price": 49.99,
            "url": "https://www.ebay.com",
            "image_urls": [
                "https://upload.wikimedia.org/wikipedia/en/thumb/5/5f/Original_Bullseye_logo.svg/1200px-Original_Bullseye_logo.svg.png",
            ],
        },
        {
            "title": f"[DEMO] {query} 1st Ed Base Set Shadowless LP/NM",
            "price": 89.95,
            "url": "https://www.ebay.com",
            "image_urls": [],
        },
    ]

# ─────────────────────────────────────────────
# FINANCIAL LOGIC
# ─────────────────────────────────────────────

PSA10_ESTIMATES = {
    # Base Set 1st Ed
    "charizard 1st": 350000, "blastoise 1st": 25000, "venusaur 1st": 12000,
    # Base Shadowless
    "charizard shadowless": 50000, "blastoise shadowless": 8000,
    # Base Unlimited
    "charizard base": 5000, "blastoise base": 1200, "venusaur base": 900,
    # Neo Genesis
    "lugia neo": 12000, "typhlosion neo": 800, "feraligatr neo": 700,
    # Neo Destiny
    "shining charizard": 8000, "shining magikarp": 3500,
    # Skyridge / Aquapolis
    "charizard skyridge": 15000, "charizard aquapolis": 8000,
    # Jungle
    "scyther jungle": 600, "clefable jungle": 700, "snorlax jungle": 1200,
    # Fossil
    "gengar fossil": 2500, "lapras fossil": 800,
}

def estimate_psa10_value(title: str) -> float:
    title_lower = title.lower()
    for key, val in PSA10_ESTIMATES.items():
        if all(k in title_lower for k in key.split()):
            return float(val)
    # Heuristic fallback
    if "1st edition" in title_lower or "1st ed" in title_lower:
        return 2000.0
    if "shadowless" in title_lower:
        return 1500.0
    if any(s in title_lower for s in ["lugia", "ho-oh", "entei", "raikou", "suicune"]):
        return 1200.0
    if "holo" in title_lower and any(s in title_lower for s in ["base set", "jungle", "fossil", "team rocket", "neo"]):
        return 800.0
    return 500.0   # conservative floor

def calculate_roi(listing_price: float, psa10_value: float,
                  psa10_prob: float, grading_fee: float = GRADING_FEE,
                  shipping: float = SHIPPING_EST) -> tuple[float, float]:
    """
    Expected net profit = (PSA10 value × P(PSA10)) − raw_price − grading − shipping.
    Also returns simple best-case net profit assuming PSA 10.
    """
    expected = (psa10_value * psa10_prob / 100.0) - listing_price - grading_fee - shipping
    best_case = psa10_value - listing_price - grading_fee - shipping
    return round(expected, 2), round(best_case, 2)

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def probability_bar(label: str, prob: float, color: str):
    st.markdown(f"""
    <div style="margin-bottom:4px;">
      <span style="font-size:0.75rem;color:#aaa;font-family:'Courier New',monospace;">{label}</span>
      <div style="background:#1a1a1a;border-radius:4px;height:14px;margin-top:2px;">
        <div style="width:{prob}%;background:{color};height:100%;border-radius:4px;
                    transition:width 0.6s ease;"></div>
      </div>
      <span style="font-size:0.7rem;color:{color};font-family:'Courier New',monospace;">{prob:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

def verdict_badge(verdict: str) -> str:
    if verdict == "PASS":
        return "🟢 **PASS** — Manual Review Recommended"
    elif verdict == "WARN":
        return "🟡 **GLINT WARNING** — Review Images Carefully"
    else:
        return "🔴 **REJECTED** — Below PSA 10 Standards"

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def render_corner_panels(patches: list, corner_names=("TL","TR","BL","BR")):
    if not patches or len(patches) < 4:
        return
    cols = st.columns(4)
    for i, (col, patch) in enumerate(zip(cols, patches[:4])):
        with col:
            label = corner_names[i] if i < len(corner_names) else f"C{i}"
            if patch.size > 0:
                pil_img = cv2_to_pil(patch)
                # Upscale for visibility
                w, h = pil_img.size
                pil_img = pil_img.resize((w*4, h*4), Image.NEAREST)
                st.image(pil_img, caption=label, use_container_width=True)

# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PSA 10 Profit Scanner",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (dark mode, mobile-first)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    background-color: #080c12 !important;
    color: #d4e1f5 !important;
}
.stApp { background: #080c12; }

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.05em;
}
.mono { font-family: 'Share Tech Mono', monospace; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1421 !important;
    border-right: 1px solid #1e3a5f;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a6fff, #005ecb);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3a7fff, #1a6fff);
    transform: translateY(-1px);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #0d1421 !important;
    color: #d4e1f5 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #0d1421 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.05rem !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: #0d1421;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 0.8rem;
}

/* Card panels */
.scan-card {
    background: #0d1421;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.scan-card.pass  { border-color: #00c87a; }
.scan-card.warn  { border-color: #f5a623; }
.scan-card.fail  { border-color: #e53e3e; }

.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: 'Share Tech Mono', monospace;
    font-weight: bold;
    margin-right: 4px;
}
.tag-pass { background: #003d24; color: #00c87a; border: 1px solid #00c87a; }
.tag-warn { background: #3d2600; color: #f5a623; border: 1px solid #f5a623; }
.tag-fail { background: #3d0000; color: #e53e3e; border: 1px solid #e53e3e; }

.hero-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2rem, 7vw, 4rem);
    font-weight: 800;
    letter-spacing: 0.08em;
    line-height: 1;
    background: linear-gradient(135deg, #4da6ff 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #4a7fc1;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 1.5rem 0;
}

/* Progress bar overrides */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1a6fff, #a78bfa) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<div class="hero-title">PSA 10 PROFIT SCANNER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Pokémon Card AI Pre-Grader &amp; ROI Engine — v1.0</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Tabs ───
tab_scan, tab_gap, tab_single, tab_config = st.tabs([
    "🔍 LIVE SCAN", "💰 GAP FINDER", "🔬 SINGLE CARD", "⚙️ CONFIG"
])

# ═══════════════════════════════════════════
# TAB 1 — LIVE SCAN
# ═══════════════════════════════════════════
with tab_scan:
    st.markdown("### eBay Live Scanner")
    st.caption("Searches eBay for raw cards within your budget and auto-grades every photo.")

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        search_query = st.text_input("Search Query", value="Charizard Base Set Holo",
                                      placeholder="e.g. Lugia Neo Genesis Holo")
    with c2:
        min_price = st.number_input("Min $", value=1, min_value=1, max_value=999)
    with c3:
        max_price = st.number_input("Max $", value=100, min_value=1, max_value=10000)
    with c4:
        num_results = st.number_input("Listings", value=10, min_value=1, max_value=50)

    psa10_filter = st.number_input("Min PSA 10 Value ($)", value=1000, min_value=0,
                                    help="Only show cards where estimated PSA 10 value exceeds this")

    run_scan = st.button("▶ RUN SCAN", use_container_width=True)

    if run_scan:
        with st.spinner("Fetching eBay listings…"):
            listings_raw = search_ebay_listings(search_query, min_price, max_price, num_results)

        if not listings_raw:
            st.warning("No listings returned. Check your search query or API credentials.")
        else:
            st.info(f"Analysing {len(listings_raw)} listing(s)… This may take a minute.")
            results: list[ListingResult] = []
            prog = st.progress(0.0)
            status = st.empty()

            for i, raw in enumerate(listings_raw):
                status.text(f"Grading listing {i+1}/{len(listings_raw)}: {raw['title'][:60]}…")
                prog.progress((i) / len(listings_raw))

                lr = ListingResult(
                    title=raw["title"],
                    price=raw["price"],
                    url=raw["url"],
                    image_urls=raw["image_urls"],
                )
                lr.psa10_value = estimate_psa10_value(lr.title)

                if lr.psa10_value < psa10_filter:
                    lr.verdict = "LOW_VALUE"
                    lr.grading = GradingResult()
                    lr.grading.reasoning = [f"PSA 10 estimated value ${lr.psa10_value:,.0f} below threshold ${psa10_filter:,.0f}"]
                    results.append(lr)
                    continue

                lr.grading = run_full_grading(lr.image_urls)

                exp_profit, best_profit = calculate_roi(lr.price, lr.psa10_value, lr.grading.psa10_prob)
                lr.net_profit = best_profit
                lr.roi_pct = ((best_profit / max(lr.price, 0.01)) * 100) if lr.price > 0 else 0

                if lr.grading.overall_pass:
                    lr.verdict = "WARN" if lr.grading.glint_flag else "PASS"
                else:
                    lr.verdict = "FAIL"

                results.append(lr)

            prog.progress(1.0)
            status.empty()

            # Sort: PASS first, then WARN, then FAIL, by net profit desc
            order = {"PASS": 0, "WARN": 1, "FAIL": 2, "LOW_VALUE": 3}
            results.sort(key=lambda r: (order.get(r.verdict, 4), -r.net_profit))

            # Summary metrics
            n_pass = sum(1 for r in results if r.verdict in ("PASS", "WARN"))
            n_fail = sum(1 for r in results if r.verdict == "FAIL")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Listings Scanned", len(results))
            m2.metric("Passed", n_pass, delta="Worth reviewing")
            m3.metric("Rejected", n_fail)
            m4.metric("Rejection Rate", f"{n_fail/max(len(results),1)*100:.0f}%")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### Results Feed")

            for r in results:
                v = r.verdict
                css_cls = "pass" if v == "PASS" else ("warn" if v == "WARN" else "fail")

                with st.expander(
                    f"{'✅' if v=='PASS' else '⚠️' if v=='WARN' else '❌'} "
                    f"{r.title[:65]} — ${r.price:.2f}",
                    expanded=(v in ("PASS", "WARN")),
                ):
                    st.markdown(f'<div class="scan-card {css_cls}">', unsafe_allow_html=True)

                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**{verdict_badge(v)}**")
                        st.markdown(f"[🔗 View on eBay]({r.url})")

                        if r.image_urls:
                            st.image(r.image_urls[0], width=180)

                    with col_b:
                        st.markdown("**Financials**")
                        st.markdown(f"""
                        <div class="mono" style="font-size:0.78rem;line-height:1.7;">
                        Raw Price:   <b>${r.price:.2f}</b><br>
                        PSA10 Est:   <b>${r.psa10_value:,.0f}</b><br>
                        Grading:     <b>−${GRADING_FEE:.0f}</b><br>
                        Shipping:    <b>−${SHIPPING_EST:.0f}</b><br>
                        <span style="color:#00c87a;">Net (Best): <b>${r.net_profit:,.0f}</b></span><br>
                        ROI:         <b>{r.roi_pct:.0f}%</b>
                        </div>
                        """, unsafe_allow_html=True)

                    if r.grading:
                        gr = r.grading
                        st.markdown("**Grade Probability**")
                        probability_bar("PSA 10", gr.psa10_prob, "#00c87a")
                        probability_bar("PSA  9", gr.psa9_prob,  "#f5a623")
                        probability_bar("PSA  8", gr.psa8_prob,  "#e06b00")
                        probability_bar("PSA ≤7", gr.psa7_below_prob, "#e53e3e")

                        if gr.reasoning:
                            st.markdown("**AI Reasoning Report**")
                            for note in gr.reasoning:
                                icon = "✓" if "✓" in note or "passed" in note.lower() else "⚠"
                                st.markdown(f"`{icon}` {note}")

                        if gr.corner_patches:
                            st.markdown("**Corner Analysis (4× Zoom)**")
                            render_corner_panels(gr.corner_patches)

                        if gr.centering_front_lr > 0:
                            st.markdown(
                                f"**Centering** — Front L/R: `{gr.centering_front_lr:.1%}` "
                                f"| T/B: `{gr.centering_front_tb:.1%}`"
                            )

                    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2 — GAP FINDER
# ═══════════════════════════════════════════
with tab_gap:
    st.markdown("### 💰 High-ROI Gap Finder")
    st.caption("Cards where the raw-to-PSA 10 price spread is historically massive. "
               "These are your highest-potential targets.")

    for entry in HIGH_ROI_SETS:
        with st.expander(f"📦 {entry['set']}", expanded=False):
            st.markdown(f"*{entry['note']}*")
            st.markdown("**Target Cards:**")
            for card in entry["cards"]:
                query_link = f"https://www.ebay.com/sch/i.html?_nkw={card.replace(' ','+')}+{entry['set'].split()[0].replace(' ','+').lower()}+pokemon+raw+ungraded&_sacat=183454"
                st.markdown(f"- [{card}]({query_link}) — [Search eBay 🔗]({query_link})")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Raw vs PSA 10 Price Reference")

    table_data = {
        "Card": [],
        "Raw (approx)": [],
        "PSA 10 (approx)": [],
        "Spread": [],
    }
    sample_cards = [
        ("Charizard 1st Ed Base", 8000, 350000),
        ("Charizard Shadowless", 1200, 50000),
        ("Charizard Base Unlimited", 200, 5000),
        ("Lugia Neo Genesis", 150, 12000),
        ("Shining Charizard Neo Destiny", 300, 8000),
        ("Gengar Fossil Holo", 80, 2500),
        ("Scyther Jungle Holo", 30, 600),
        ("Ho-Oh Neo Revelation", 90, 3000),
    ]
    for name, raw, psa10 in sample_cards:
        table_data["Card"].append(name)
        table_data["Raw (approx)"].append(f"${raw:,}")
        table_data["PSA 10 (approx)"].append(f"${psa10:,}")
        table_data["Spread"].append(f"${psa10-raw:,} ({(psa10/raw):.0f}×)")

    import pandas as pd
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# TAB 3 — SINGLE CARD ANALYSER
# ═══════════════════════════════════════════
with tab_single:
    st.markdown("### 🔬 Single Card Image Analyser")
    st.caption("Paste an image URL or upload a photo to run the full CV pre-grade pipeline.")

    input_mode = st.radio("Input Method", ["Upload Image", "Paste URL"], horizontal=True)

    img_for_analysis = None
    img_url_list = []

    if input_mode == "Upload Image":
        uploaded = st.file_uploader("Upload card photo (JPG/PNG)", type=["jpg","jpeg","png"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            img_for_analysis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        url_input = st.text_input("Image URL", placeholder="https://i.ebayimg.com/...")
        if url_input.strip():
            img_url_list = [url_input.strip()]

    card_title_single = st.text_input("Card Name (for ROI estimate)",
                                       placeholder="e.g. Charizard Base Set Holo Rare")
    raw_price_single  = st.number_input("Raw Listing Price ($)", value=50.0, min_value=0.0)

    run_single = st.button("▶ ANALYSE CARD", use_container_width=True)

    if run_single:
        if img_for_analysis is None and not img_url_list:
            st.error("Please upload an image or paste a URL.")
        else:
            with st.spinner("Running CV analysis pipeline…"):
                if img_for_analysis is not None:
                    # Wrap locally-loaded image in a mock grading flow
                    gr = GradingResult()
                    res_pass, res_px = check_resolution(img_for_analysis)
                    gr.resolution_px = res_px
                    if not res_pass:
                        gr.resolution_flag = True
                        gr.reasoning.append(f"Low resolution: {res_px[0]}×{res_px[1]}px")
                    else:
                        lr, tb, cf, cr = check_centering(img_for_analysis)
                        gr.centering_front_lr, gr.centering_front_tb = lr, tb
                        gr.centering_flag = cf
                        gr.reasoning.extend(cr)

                        cor_flag, cor_labels, patches = check_corners_whitening(img_for_analysis)
                        gr.corners_flag = cor_flag
                        gr.corner_patches = patches
                        gr.reasoning.extend(cor_labels)

                        g_flag, s_flag, g_reasons = check_glint_and_scratches(img_for_analysis)
                        gr.glint_flag = g_flag
                        gr.scratch_flag = s_flag
                        gr.reasoning.extend(g_reasons)

                    hard_fails = [gr.centering_flag, gr.corners_flag, gr.scratch_flag, gr.resolution_flag]
                    gr.overall_pass = not any(hard_fails)
                    if gr.overall_pass:
                        gr.reasoning.append("✓ All automated checks passed")
                    gr = compute_grade_probabilities(gr)
                else:
                    gr = run_full_grading(img_url_list)

            psa10_val = estimate_psa10_value(card_title_single)
            _, best_profit = calculate_roi(raw_price_single, psa10_val, gr.psa10_prob)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            # Show uploaded image
            if img_for_analysis is not None:
                st.image(cv2_to_pil(img_for_analysis), caption="Input Image", width=280)

            # Verdict
            v = "PASS" if gr.overall_pass else "FAIL"
            if gr.overall_pass and gr.glint_flag:
                v = "WARN"
            st.markdown(f"## {verdict_badge(v)}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Grade Probability")
                probability_bar("PSA 10", gr.psa10_prob, "#00c87a")
                probability_bar("PSA  9", gr.psa9_prob,  "#f5a623")
                probability_bar("PSA  8", gr.psa8_prob,  "#e06b00")
                probability_bar("PSA ≤7", gr.psa7_below_prob, "#e53e3e")

            with col2:
                st.markdown("#### Financials")
                st.markdown(f"""
                <div class="mono" style="font-size:0.85rem;line-height:2;">
                Raw Price:     <b>${raw_price_single:.2f}</b><br>
                PSA 10 Est:    <b>${psa10_val:,.0f}</b><br>
                Grading Fee:   <b>−${GRADING_FEE:.0f}</b><br>
                Shipping Est:  <b>−${SHIPPING_EST:.0f}</b><br>
                <span style="color:#00c87a;font-size:1.1rem;">Best-Case Net: <b>${best_profit:,.0f}</b></span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### AI Reasoning Report")
            for note in gr.reasoning:
                icon = "✓" if ("✓" in note or "passed" in note.lower()) else "⚠"
                st.markdown(f"`{icon}` {note}")

            st.markdown(f"**Resolution:** `{gr.resolution_px[0]}×{gr.resolution_px[1]}px`")
            st.markdown(f"**Centering:** Front L/R `{gr.centering_front_lr:.1%}` | T/B `{gr.centering_front_tb:.1%}`")

            if gr.corner_patches:
                st.markdown("#### Corner Analysis (4× Zoom)")
                render_corner_panels(gr.corner_patches)

# ═══════════════════════════════════════════
# TAB 4 — CONFIG
# ═══════════════════════════════════════════
with tab_config:
    st.markdown("### ⚙️ API Configuration")
    st.caption("Configure your eBay API keys. These are stored in `config.yaml` or as environment variables.")

    st.code("""
# config.yaml — place in the same directory as app.py

ebay:
  app_id:  "YourEbayAppID-here"
  cert_id: "YourEbayCertID-here"
  dev_id:  "YourEbayDevID-here"
""", language="yaml")

    st.markdown("**Streamlit Cloud Secrets (alternative to config.yaml):**")
    st.code("""
# .streamlit/secrets.toml (for cloud deploy)

[ebay]
app_id  = "YourEbayAppID-here"
cert_id = "YourEbayCertID-here"
dev_id  = "YourEbayDevID-here"
""", language="toml")

    st.markdown("---")
    st.markdown("### 🚀 Free Deployment to Streamlit Cloud")
    st.markdown("""
1. **Push to GitHub** — Create a public (or private) repo with these files:
   ```
   app.py
   requirements.txt
   config.yaml        ← or use Streamlit Secrets instead
   ```

2. **Sign up at** [share.streamlit.io](https://share.streamlit.io) (free)

3. **New App** → connect your GitHub repo → set `app.py` as the main file

4. **Add Secrets** in the Streamlit Cloud dashboard:
   - Go to your app → ⋮ menu → **Settings** → **Secrets**
   - Paste your eBay keys in TOML format (see above)

5. **Deploy** — your app will be live at a public URL accessible from your phone.

6. **eBay Developer Keys:**
   - Register at [developer.ebay.com](https://developer.ebay.com)
   - Create an app → get **App ID (Client ID)**, **Cert ID (Client Secret)**, **Dev ID**
   - Use **Production** keys (not Sandbox) for live listings
    """)

    st.markdown("---")
    st.markdown("### 📦 Grading Logic Reference")
    st.markdown(f"""
| Check | Threshold | On Fail |
|---|---|---|
| Resolution | Min `{MIN_IMAGE_PX}px` each dimension | Hard Reject |
| Front Centering L/R | Max `{CENTER_THRESH:.0%}`/`{1-CENTER_THRESH:.0%}` | Hard Reject |
| Front Centering T/B | Max `{CENTER_THRESH:.0%}`/`{1-CENTER_THRESH:.0%}` | Hard Reject |
| Corner Whitening | ≥`{WHITE_CLUSTER}` white pixels in corner patch | Hard Reject |
| Scratch in Glint | >`{SCRATCH_THRESH:.1%}` edge density in overexposed zone | Hard Reject |
| Glint Only (no scratches) | >`2%` holo area overexposed | Warning Only |
    """)

    st.markdown("### 🎯 Strictness Philosophy")
    st.info(
        "This scanner is calibrated to **reject aggressively**. A card that passes all checks "
        "still needs your manual visual inspection before purchase. The AI provides a *floor* "
        "assessment — not a guarantee. PSA grading has inherent subjectivity that no CV model "
        "can fully replicate. Always buy from sellers with clear, high-resolution, multi-angle photos."
    )

# ─── Footer ───
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:\'Share Tech Mono\',monospace;'
    'font-size:0.7rem;color:#2a4a6e;">'
    'PSA 10 PROFIT SCANNER • For educational use only • '
    'Not financial advice • Always verify manually before purchasing'
    '</div>',
    unsafe_allow_html=True
)
