from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_MASTER = PROJECT_ROOT / "data" / "master"

SQUIGGLE_BASE_URL = "https://api.squiggle.com.au/"
SQUIGGLE_USER_AGENT = "AFL-Prediction-Model/0.1 (github.com/ollie/afl-model)"

AFLTABLES_BASE_URL = "https://afltables.com/afl/"
FOOTYWIRE_BASE_URL = "https://www.footywire.com/afl/footy/"

SCRAPE_DELAY_SECONDS = 2.0

FIRST_YEAR = 2012
CURRENT_YEAR = 2026
