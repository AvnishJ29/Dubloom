"""Start the Dubloom studio: `python main.py` (or ./run.sh)."""

import uvicorn

from backend.config import settings
from backend.logging_setup import configure_logging

configure_logging()
from backend.api import app  # noqa: E402  (logging must be configured first)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info", access_log=False)
