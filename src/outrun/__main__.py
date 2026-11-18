"""Entry point: ``python -m outrun <command>`` (also installed as the ``outrun`` script).

The real command-line interface lives in ``outrun.cli`` (spec 14, CLI-* requirements). This module only
forwards to it so that ``python -m outrun`` and the console script behave identically.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run the Outrun CLI and return its exit code (forwards to ``outrun.cli.main``)."""
    from outrun.cli import main as cli_main

    return int(cli_main(argv) or 0)


if __name__ == "__main__":
    sys.exit(main())
