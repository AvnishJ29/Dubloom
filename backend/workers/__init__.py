"""Command-line workers that the studio runs as subprocesses.

Most run in their own virtualenvs (.venv-latentsync, .venv-diarizen, .venv-restore, ...) so their
pinned model stacks stay out of the studio process, and import nothing from it. They report
progress on stdout as PROGRESS / Rendered / DONE lines that the callers parse.
"""
