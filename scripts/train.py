"""Train the search relevance model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hd_search_relevance.cli import main

if __name__ == "__main__":
    cli_args = sys.argv[1:]
    sys.argv = ["hd-search", "train", *cli_args]
    main()
