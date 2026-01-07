from __future__ import annotations

import logging
import sys

from femto.logger import CustomConsoleFormatter

# from femto.logger import DEBUGFORMATTER

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_ch1 = logging.StreamHandler(sys.stdout)
_ch1.setLevel(logging.INFO)
_ch1.setFormatter(CustomConsoleFormatter())
_ch1.addFilter(lambda record: record.levelno <= logging.INFO)
logger.addHandler(_ch1)
_ch2 = logging.StreamHandler(sys.stdout)
_ch2.setLevel(logging.WARNING)
_ch2.setFormatter(CustomConsoleFormatter())
logger.addHandler(_ch2)

# _fh = logging.FileHandler('log.log', mode='w')
# _fh.setLevel(logging.DEBUG)
# _fh.setFormatter(logging.Formatter(DEBUGFORMATTER, '%Y-%m-%d %H:%M:%S'))
# logger.addHandler(_fh)
