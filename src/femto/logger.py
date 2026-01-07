from __future__ import annotations

import logging
from datetime import datetime


class CustomConsoleFormatter(logging.Formatter):
    # ANSI styles
    RESET = '\x1b[0m'
    BOLD = '\x1b[1m'
    DIM = '\x1b[2m'
    ITALIC = '\x1b[3m'
    REVERSE = '\x1b[7m'

    # Color palette
    FG_WHITE = '\x1b[38;2;236;239;244m'
    FG_GRAY = '\x1b[38;2;160;170;180m'
    FG_BLUE = '\x1b[38;2;136;192;208m'
    FG_GREEN = '\x1b[38;2;163;190;140m'
    FG_YELLOW = '\x1b[38;2;235;203;139m'
    FG_RED = '\x1b[38;2;191;97;106m'
    FG_PURPLE = '\x1b[38;2;180;142;173m'

    LEVEL_STYLES = {
        logging.DEBUG: FG_GRAY,
        logging.INFO: FG_GREEN,
        logging.WARNING: FG_YELLOW,
        logging.ERROR: FG_RED,
        logging.CRITICAL: BOLD + REVERSE + FG_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        level_style = self.LEVEL_STYLES.get(record.levelno, self.FG_WHITE)
        level = f"{level_style}{record.levelname:<8}{self.RESET}"

        module = f"{self.FG_BLUE}{record.module}{self.RESET}"
        message = f"{self.FG_WHITE}{record.getMessage()}{self.RESET}"
        timestamp = f"{self.DIM}{self.FG_GRAY}{ts}{self.RESET}"

        return f"{timestamp}  {level}  {module} : {message}"
