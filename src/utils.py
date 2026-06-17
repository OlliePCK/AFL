import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import SCRAPE_DELAY_SECONDS, DATA_RAW

_last_request_time: dict[str, float] = {}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("afl")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        fh = logging.FileHandler(DATA_RAW.parent / "collection.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _get_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


_session = _get_session()


def fetch_url(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    now = time.time()
    last = _last_request_time.get(domain, 0)
    wait = SCRAPE_DELAY_SECONDS - (now - last)
    if wait > 0:
        time.sleep(wait)
    resp = _session.get(url, params=params, headers=headers, timeout=30)
    _last_request_time[domain] = time.time()
    resp.raise_for_status()
    return resp


def ensure_dirs():
    from src.config import DATA_RAW, DATA_PROCESSED, DATA_MASTER
    for d in [DATA_RAW, DATA_PROCESSED, DATA_MASTER]:
        d.mkdir(parents=True, exist_ok=True)
