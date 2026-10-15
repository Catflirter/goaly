"""``python -m goaly``: the `goaly` command, for the solve container, where goaly is on
PYTHONPATH rather than installed (spec §6.1)."""

import sys

from goaly.cli import main

sys.exit(main())
