"""Run CPU-heavy diarization without blocking the FastAPI process."""

import json
import sys
from pathlib import Path

from backend.speakers.diarization import speaker_turns


def main(audio: Path, output: Path, num_speakers: int | None = None):
    def progress(done, total):
        print(f'PROGRESS {done} {total}', flush=True)

    turns, mode = speaker_turns(audio, progress=progress, num_speakers=num_speakers)
    output.write_text(
        json.dumps({'mode': mode, 'segments': [turn.public() for turn in turns]}), encoding='utf-8'
    )


if __name__ == '__main__':
    hint = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else None
    main(Path(sys.argv[1]), Path(sys.argv[2]), hint)
