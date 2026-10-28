"""HTTP API for creating, watching and retrieving dubbing jobs (Bodhan Indic models only).

- `server`: the FastAPI app, middleware, startup recovery and the built frontend
- `system`: sign-in, studio configuration and health
- `jobs`: creating, listing, watching, resuming, cancelling, renaming and deleting jobs
- `editing`: re-voicing, script edits, AI proposals, speaker merges and line takes
- `shorts`: vertical shorts cut from a finished dub
- `media`: the dub, source, poster, subtitles and archived versions for playback and download
- `schemas`: request bodies; `helpers`: validation, job lookup and scheduling shared by routes
"""

from backend.api.server import app

__all__ = ["app"]
