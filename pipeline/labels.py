"""Label taxonomy for the NOAA PIFSC Pacific benthic cover Tier-3 (T3) classes.

T3 -> T1 mapping was derived from the bucket itself: every crop in
`pacific_t3_splits/<split>/<T3>/` appears under exactly one `pacific_t1/<T1>/`
folder with the same file name.

Long names are a best-effort expansion of the PIFSC CoralNet short codes and
are used for zero-shot prompts and display. Entries marked `sure=False` are
guesses; correct them here (or pass --labels-csv to build.py) if you know better.
"""

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

# code: (T1 group, long name, sure)
T3 = {
    "ACAS": ("CORAL", "Acanthastrea coral", True),
    "ACBR": ("CORAL", "branching Acropora coral", True),
    "ACTA": ("CORAL", "tabulate Acropora coral", True),
    "ASPP": ("MA", "Asparagopsis red seaweed", True),
    "ASSP": ("CORAL", "Astreopora coral", True),
    "ASTS": ("CORAL", "Astrea coral", False),
    "BGMA": ("MA", "blue-green algae (cyanobacteria) mat", True),
    "BRMA": ("MA", "brown macroalgae", True),
    "CAUL": ("MA", "Caulerpa green seaweed", True),
    "CCAH": ("CCA", "crustose coralline algae on hard substrate", True),
    "CCAR": ("CCA", "crustose coralline algae on rubble", False),
    "CMOR": ("I", "corallimorph", True),
    "COSP": ("CORAL", "Coscinaraea coral", True),
    "CYPS": ("CORAL", "Cyphastrea coral", True),
    "DICO": ("MA", "Dictyosphaeria green bubble algae", False),
    "DICT": ("MA", "Dictyota brown seaweed", True),
    "DISP": ("CORAL", "Diploastrea coral", False),
    "ECHP": ("CORAL", "Echinophyllia coral", True),
    "EMA": ("MA", "encrusting macroalgae", True),
    "ENC": ("CORAL", "encrusting hard coral", False),
    "FASP": ("CORAL", "Favia coral", True),
    "FAVS": ("CORAL", "Favites coral", True),
    "FINE": ("SED", "fine sediment", True),
    "FOL": ("CORAL", "foliose hard coral", True),
    "FREE": ("CORAL", "free-living mushroom coral", True),
    "FUSP": ("CORAL", "Fungia mushroom coral", True),
    "GASP": ("CORAL", "Galaxea coral", True),
    "GOAL": ("CORAL", "Goniopora or Alveopora coral", True),
    "GONS": ("CORAL", "Goniastrea coral", True),
    "GRMA": ("MA", "green macroalgae", True),
    "HALI": ("MA", "Halimeda calcareous green algae", True),
    "HCOE": ("CORAL", "Heliopora blue coral", True),
    "HYSP": ("CORAL", "Hydnophora coral", True),
    "ISSP": ("CORAL", "Isopora coral", True),
    "LEPT": ("CORAL", "Leptastrea coral", True),
    "LOBO": ("MA", "Lobophora brown seaweed", True),
    "LOBS": ("CORAL", "Lobophyllia coral", True),
    "LPHY": ("CORAL", "Leptoria phrygia brain coral", True),
    "MICR": ("MA", "Microdictyon green net algae", True),
    "MISP": ("CORAL", "Millepora fire coral", True),
    "MOBF": ("MF", "mobile fauna", True),
    "MOBR": ("CORAL", "branching Montipora coral", True),
    "MOEN": ("CORAL", "encrusting Montipora coral", True),
    "MOFO": ("CORAL", "foliose Montipora coral", True),
    "OCTO": ("SC", "octocoral soft coral", True),
    "PADI": ("MA", "Padina brown seaweed", True),
    "PAEN": ("CORAL", "encrusting Pavona coral", True),
    "PAMA": ("CORAL", "massive Pavona coral", True),
    "PESP": ("MA", "Peyssonnelia red encrusting algae", True),
    "PHSP": ("CORAL", "PHSP hard coral", False),
    "PLSP": ("CORAL", "Platygyra brain coral", True),
    "POBR": ("CORAL", "branching Porites coral", True),
    "POCS": ("CORAL", "Pocillopora coral", True),
    "POEN": ("CORAL", "encrusting Porites coral", True),
    "POFO": ("CORAL", "foliose Porites coral", True),
    "POMA": ("CORAL", "massive Porites coral", True),
    "PSSP": ("CORAL", "Psammocora coral", True),
    "RDMA": ("MA", "red macroalgae", True),
    "SAND": ("SED", "sand", True),
    "SP": ("I", "sponge", True),
    "STYS": ("CORAL", "Stylophora coral", True),
    "TUN": ("I", "tunicate", True),
    "TURFH": ("TURF", "turf algae on hard substrate", True),
    "TURFR": ("TURF", "turf algae on rubble", True),
    "TURS": ("CORAL", "Turbinaria coral", True),
    "UPMA": ("MA", "upright macroalgae", True),
    "ZO": ("I", "zoanthid", True),
}

PROMPT_TEMPLATE = "An underwater close-up photo of {name} on a coral reef."


def group_of(code: str) -> str:
    return T3.get(code, ("UNK", code, False))[0]


def name_of(code: str) -> str:
    return T3.get(code, ("UNK", code, False))[1]


def load_overrides(path: str) -> None:
    """CSV with columns code,name[,group] to override/add entries."""
    import csv

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            code = row["code"].strip()
            grp = row.get("group") or group_of(code)
            T3[code] = (grp, row["name"].strip(), True)
