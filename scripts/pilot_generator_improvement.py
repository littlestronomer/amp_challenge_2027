"""Run the raw-yield generator pilot (RAFT by default, or --method grpo)."""
import sys

from pilot_grpo import main

if __name__ == "__main__":
    main(["--method", "raft", "--raw-evaluation", *sys.argv[1:]])
