"""Run the lane CLI: python -m goaly.rsi lane --lane effisql.

The process exits with the CLI's code as soon as it returns: an implementer thread still waiting
on a model call or a smoke run after the lane stopped must not hold the exit (spec §8.4 step 6),
so the exit skips the interpreter's join of worker threads, once the logs are flushed.
"""

import logging
import os
import sys

from goaly.rsi.cli import main

code = main()
logging.shutdown()
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
