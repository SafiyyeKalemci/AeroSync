"""Backward-compatible entry point for the official integration runner."""

from competition.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
