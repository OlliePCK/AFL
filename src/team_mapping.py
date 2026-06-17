# Canonical team names match Squiggle's naming.
# Maps every known variant from AFL Tables, Footywire, and other sources.

TEAM_NAME_MAP = {
    # Squiggle canonical names (identity mappings)
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane Lions",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Greater Western Sydney": "Greater Western Sydney",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",

    # AFL Tables variants
    "Adelaide Crows": "Adelaide",
    "Brisbane": "Brisbane Lions",
    "Brisbane Bears": "Brisbane Lions",
    "Fremantle Dockers": "Fremantle",
    "Geelong Cats": "Geelong",
    "Gold Coast Suns": "Gold Coast",
    "GWS Giants": "Greater Western Sydney",
    "GWS": "Greater Western Sydney",
    "Greater Western Sydney Giants": "Greater Western Sydney",
    "Hawthorn Hawks": "Hawthorn",
    "Melbourne Demons": "Melbourne",
    "North Melbourne Kangaroos": "North Melbourne",
    "Kangaroos": "North Melbourne",
    "Port Adelaide Power": "Port Adelaide",
    "Richmond Tigers": "Richmond",
    "St Kilda Saints": "St Kilda",
    "Sydney Swans": "Sydney",
    "West Coast Eagles": "West Coast",
    "Western Bulldogs Dogs": "Western Bulldogs",
    "Footscray": "Western Bulldogs",
    "Bulldogs": "Western Bulldogs",

    # Footywire variants
    "Collingwood Magpies": "Collingwood",
    "Carlton Blues": "Carlton",
    "Essendon Bombers": "Essendon",
}

# Squiggle team IDs
SQUIGGLE_TEAM_IDS = {
    "Adelaide": 1, "Brisbane Lions": 2, "Carlton": 3, "Collingwood": 4,
    "Essendon": 5, "Fremantle": 6, "Geelong": 7, "Gold Coast": 8,
    "Greater Western Sydney": 9, "Hawthorn": 10, "Melbourne": 11,
    "North Melbourne": 12, "Port Adelaide": 13, "Richmond": 14,
    "St Kilda": 15, "Sydney": 16, "West Coast": 17, "Western Bulldogs": 18,
}


def normalize_team(name: str) -> str:
    name = name.strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    # Try case-insensitive match
    for key, val in TEAM_NAME_MAP.items():
        if key.lower() == name.lower():
            return val
    raise ValueError(f"Unknown team name: '{name}'. Add it to TEAM_NAME_MAP.")
