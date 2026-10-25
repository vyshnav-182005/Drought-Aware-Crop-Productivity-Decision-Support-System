"""Canonical district name mapping, per Project_Implementation_Plan.md Section 5."""

CANONICAL_DISTRICTS = [
    "Ahmednagar",   # canonical: Ahmednagar/Ahilyanagar
    "Beed",
    "Osmanabad",    # canonical: Osmanabad/Dharashiv
    "Latur",
    "Solapur",
]

# Maps any source-specific spelling -> canonical name
DISTRICT_ALIASES = {
    "ahilyanagar": "Ahmednagar",
    "ahmadnagar": "Ahmednagar",
    "ahmednagar": "Ahmednagar",
    "beed": "Beed",
    "bid": "Beed",
    "dharashiv": "Osmanabad",
    "osmanabad": "Osmanabad",
    "latur": "Latur",
    "solapur": "Solapur",
}

# Contingency plan PDF filename -> canonical district (source of truth for this pipeline)
CONTINGENCY_PDF_DISTRICT = {
    "Maharashtra 27-Beed-3 1-12-2011.pdf": "Beed",
    "Maharashtra 28 -Latur- 31-12-2011.pdf": "Latur",
    "Maharashtra 30-Osmanabad- 31-12-2011.pdf": "Osmanabad",
    "MH1-Solapur 3.2.2011.pdf": "Solapur",
    "MH2-AHMEDNAGAR 31.03.2011.pdf": "Ahmednagar",
}

CROP_ALIASES = {
    "pearlmillet": "Bajra",
    "pearl millet": "Bajra",
    "bajra": "Bajra",
    "sorghum": "Jowar",
    "jowar": "Jowar",
    "cotton": "Cotton",
    "pigeonpea": "Tur",
    "pigeon pea": "Tur",
    "tur": "Tur",
    "arhar": "Tur",
    "soybean": "Soybean",
    "soyabean": "Soybean",
    "wheat": "Wheat",
    "gram": "Gram",
    "chickpea": "Gram",
    "safflower": "Safflower",
    "sunflower": "Sunflower",
    "sugarcane": "Sugarcane",
    "maize": "Maize",
    "ginger": "Ginger",
    "turmeric": "Turmeric",
}


def normalize_district(name: str) -> str:
    key = name.strip().lower()
    return DISTRICT_ALIASES.get(key, name.strip().title())


def normalize_crop(name: str) -> str:
    key = name.strip().lower()
    # strip trailing variety parentheticals like "Sorghum (Nilwa, MP ...)"
    key = key.split("(")[0].strip()
    return CROP_ALIASES.get(key, name.strip().title())

# Extended crop aliases - fallback/minor crops that appear in contingency tables
# but are not modeling targets (per Project_Implementation_Plan.md Section 2.2).
CROP_ALIASES.update({
    "black gram": "Black gram",
    "blackgram": "Black gram",
    "green gram": "Green gram",
    "greengram": "Green gram",
    "cowpea": "Cowpea",
    "moth bean": "Moth bean",
    "mothbean": "Moth bean",
    "horse gram": "Horse gram",
    "horsegram": "Horse gram",
    "niger": "Niger",
    "castor": "Castor",
    "sesamum": "Sesamum",
    "til": "Sesamum",
    "ginger": "Ginger",
    "turmeric": "Turmeric",
    "maize": "Maize",
    "grape": "Grape",
    "mango": "Mango",
    "sweet orange": "Sweet orange",
    "mosambi": "Sweet orange",
    "kagzi lime": "Kagzi lime",
    "kagzilime": "Kagzi lime",
    "onion": "Onion",
    "chilli": "Chilli",
    "garlic": "Garlic",
    "banana": "Banana",
    "custard apple": "Custard apple",
    "ber": "Ber",
    "pomegranate": "Pomegranate",
    "guava": "Guava",
    "gauva": "Guava",
    "brinjal": "Brinjal",
    "tomato": "Tomato",
    "okra": "Okra",
    "bhendi": "Okra",
    "cabbage": "Cabbage",
    "cauliflower": "Cauliflower",
    "sapota": "Sapota",
    # spelling variants seen in the plans
    "pigeoenpea": "Tur",
    "pomogranate": "Pomegranate",
    "grapes": "Grape",
    "groundnut": "Groundnut",
    "linseed": "Linseed",
    "sunflowers": "Sunflower",
})

# Canonical crop names this pipeline recognizes - used to reject noise cells
# (measurement instructions, soil-type text, stray table fragments) that
# pdfplumber's table extraction occasionally misaligns into the crop column.
CROP_VOCAB = set(CROP_ALIASES.values())


def split_and_normalize_crops(raw: str) -> list:
    """Split a raw crop cell on '/', ',' or '+' and keep only recognized crops.

    Contingency-plan tables often list combinations like "Pearl millet +
    Pigeonpea" or "Turmeric / Ginger" in one cell. Returns a list (possibly
    empty) of canonical crop names actually found in the cell.
    """
    if not raw:
        return []
    raw = raw.replace("&", "/")
    parts = re.split(r"[/,+]| and ", raw)
    found = []
    for part in parts:
        part = part.strip()
        # strip leading season words that sometimes prefix a crop name
        part = re.sub(r"^(kharif|rabi|irrigated)\s+", "", part, flags=re.I)
        part = part.split("(")[0].strip()
        key = part.lower()
        if key in CROP_ALIASES:
            name = CROP_ALIASES[key]
            if name not in found:
                found.append(name)
    return found


import re  # noqa: E402
