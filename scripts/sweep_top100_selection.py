"""P0 per-library scaling versus P1 fixed seed-42 hybrid scaling; cached CPU experiment.

Includes all P0 diagnostics, so running analyze_selection.py separately is optional.
Does not enable a safety-weight sweep or change production defaults.
"""

from analyze_selection import parser, run


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    cli.description = __doc__
    run(cli.parse_args(argv), policies=("P0", "P1"))


if __name__ == "__main__":
    main()
