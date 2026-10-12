"""Label taxonomy for the NOAA PIFSC Pacific benthic cover Tier-3 (T3) classes.

Official names and the T1 -> T2 -> T2b -> T3 hierarchy come from the PIFSC taxonomy
tree in `taxonomy_tree.json` next to this file:

    T1   broad group           CORAL, MA, TURF, CCA, SED, I, SC, MF, ...
    T2   growth form / type    ENC (encrusting hard coral), BR, TAB, MASS, UPMA, EMA, ...
    T2b  genus / subgroup      POSP (Porites sp), MOSP (Montipora sp), BRMA, ...
    T3   class                 the 67 codes the dataset is split by

`PROMPT_NAMES` holds a plain-English phrase per class for the zero-shot prompts; it
is checked against the official names but worded for an embedding model (no "sp",
no "CCA" abbreviation, a hint of colour or shape where that is what the code means).
Codes missing from the tree fall back to their own code and are marked `sure=False`.
"""
import json
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("taxonomy_tree.json")

T1_GROUPS = {
    # code: (display name, prompt phrase); order = palette slot order
    "CORAL": ("Hard coral", "hard coral"),
    "MA": ("Macroalgae", "macroalgae seaweed"),
    "TURF": ("Turf algae", "turf algae"),
    "CCA": ("Crustose coralline algae", "pink crustose coralline algae"),
    "SED": ("Sediment", "sand or sediment"),
    "I": ("Sessile invertebrates", "a sessile invertebrate"),
    "SC": ("Soft coral", "soft coral"),
    "MF": ("Mobile fauna", "a mobile animal such as a fish, urchin or sea cucumber"),
}

PROMPT_NAMES = {
    "ACAS": "encrusting Acanthastrea coral",
    "ACBR": "branching Acropora coral",
    "ACTA": "tabulate Acropora table coral",
    "ASPP": "Asparagopsis red seaweed",
    "ASSP": "encrusting Astreopora coral",
    "ASTS": "massive Astrea coral",
    "BGMA": "blue-green algae (cyanobacteria) mat",
    "BRMA": "brown macroalgae",
    "CAUL": "Caulerpa green seaweed",
    "CCAH": "crustose coralline algae on hard substrate",
    "CCAR": "crustose coralline algae on rubble",
    "CMOR": "corallimorph",
    "COSP": "encrusting Coscinaraea coral",
    "CYPS": "encrusting Cyphastrea coral",
    "DICO": "Dictyopteris or Dictyota brown seaweed",
    "DICT": "Dictyosphaeria green bubble algae",
    "DISP": "massive Diploastrea coral",
    "ECHP": "encrusting Echinopora coral",
    "EMA": "encrusting macroalgae",
    "ENC": "encrusting hard coral",
    "FASP": "massive Favia coral",
    "FAVS": "massive Favites coral",
    "FINE": "fine sediment",
    "FOL": "foliose hard coral",
    "FREE": "free-living mushroom coral",
    "FUSP": "Fungia mushroom coral",
    "GASP": "Galaxea coral",
    "GOAL": "Goniopora or Alveopora coral",
    "GONS": "massive Goniastrea coral",
    "GRMA": "green macroalgae",
    "HALI": "Halimeda calcareous green algae",
    "HCOE": "Heliopora blue coral",
    "HYSP": "massive Hydnophora coral",
    "ISSP": "Isopora coral",
    "LEPT": "encrusting Leptastrea coral",
    "LOBO": "Lobophora brown seaweed",
    "LOBS": "massive Lobophyllia coral",
    "LPHY": "Leptoria brain coral",
    "MICR": "Microdictyon green net algae",
    "MISP": "Millepora fire coral",
    "MOBF": "mobile fauna",
    "MOBR": "branching Montipora coral",
    "MOEN": "encrusting Montipora coral",
    "MOFO": "foliose Montipora coral",
    "OCTO": "octocoral soft coral",
    "PADI": "Padina brown seaweed",
    "PAEN": "encrusting Pavona coral",
    "PAMA": "massive Pavona coral",
    "PESP": "Peyssonnelia red encrusting algae",
    "PHSP": "encrusting Phymastrea coral",
    "PLSP": "Platygyra brain coral",
    "POBR": "branching Porites coral",
    "POCS": "Pocillopora coral",
    "POEN": "encrusting Porites coral",
    "POFO": "foliose Porites coral",
    "POMA": "massive Porites coral",
    "PSSP": "encrusting Psammocora coral",
    "RDMA": "red macroalgae",
    "SAND": "sand",
    "SP": "sponge",
    "STYS": "Stylophora coral",
    "TUN": "tunicate",
    "TURFH": "turf algae on hard substrate",
    "TURFR": "turf algae on rubble",
    "TURS": "Turbinaria coral",
    "UPMA": "upright macroalgae",
    "ZO": "zoanthid",
}

PROMPT_TEMPLATE = "An underwater close-up photo of {name} on a coral reef."


def _split(node_name: str) -> tuple[str, str]:
    code, _, name = node_name.partition(" - ")
    return code.strip(), (name or code).strip()


def load_tree(path: Path = TAXONOMY_PATH) -> dict[str, dict]:
    """code -> {t1, t2, t2_name, t2b, t2b_name, name} for every leaf in the tree."""
    out: dict[str, dict] = {}

    def walk(node: dict, path: dict) -> None:
        tier = node["tier"]
        if tier == "CODE":
            out[node["name"]] = path
            return
        if tier != "TIER0":
            code, name = _split(node["name"])
            key = {"TIER_1": "t1", "TIER_2": "t2", "TIER_2b": "t2b", "TIER_3": "t3"}[tier]
            path = {**path, key: code, key + "_name": name}
        for child in node.get("children", []):
            walk(child, path)

    walk(json.loads(path.read_text()), {})
    return {code: {"t1": p["t1"], "t2": p["t2"], "t2_name": p["t2_name"], "t2b": p["t2b"],
                   "t2b_name": p["t2b_name"], "name": p["t3_name"]} for code, p in out.items()}


TAXONOMY = load_tree()
_overrides: dict[str, dict] = {}


def info(code: str) -> dict:
    """Taxonomy entry for a class, with --labels-csv overrides applied. `sure` is False for
    codes the tree does not know (their name is just the code)."""
    base = TAXONOMY.get(code) or {"t1": "UNK", "t2": code, "t2_name": code, "t2b": code,
                                  "t2b_name": code, "name": code}
    return {**base, "sure": code in TAXONOMY or code in _overrides,
            "prompt": PROMPT_NAMES.get(code, base["name"]), **_overrides.get(code, {})}


def group_of(code: str) -> str:
    return info(code)["t1"]


def name_of(code: str) -> str:
    return info(code)["name"]


def prompt_of(code: str) -> str:
    return info(code)["prompt"]


def load_overrides(path: str) -> None:
    """CSV with columns code,name[,group] to override/add entries. The name is used for
    display and for the zero-shot prompt."""
    import csv

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code, name = row["code"].strip(), row["name"].strip()
            o = {"name": name, "prompt": name}
            if row.get("group"):
                o["t1"] = row["group"].strip()
            _overrides[code] = o
