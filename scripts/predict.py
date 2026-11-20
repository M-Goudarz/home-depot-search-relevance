"""Run batch prediction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hd_search_relevance.cli import main

if __name__ == "__main__":
    cli_args = sys.argv[1:]
    sys.argv = ["hd-search", "predict", *cli_args]
    main()
