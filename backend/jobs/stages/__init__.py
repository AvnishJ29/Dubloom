"""The pipeline's stages, one module each, in the order `process_job` runs them:

source -> speakers -> (background work starts) -> script -> voicing -> mix -> render.

Every stage takes the run's `DubRun` (backend/jobs/context.py), reads what earlier stages left
there, and fills in its own results.
"""
