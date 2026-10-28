"""Download the official public sherpa models used by local diarization.

Run from the project root: python -m backend.speakers.setup_local_models
"""

import tarfile
from urllib.request import urlretrieve
from backend.config import settings

BASE = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/'


def install_models():
    folder = settings.jobs_dir / 'models' / 'sherpa'
    folder.mkdir(parents=True, exist_ok=True)
    segmentation = folder / 'sherpa-onnx-pyannote-segmentation-3-0/model.onnx'
    if not segmentation.is_file():
        archive = folder / 'segmentation.tar.bz2'
        urlretrieve(
            BASE + 'speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2',
            archive,
        )
        with tarfile.open(archive) as source:
            source.extractall(folder, filter='data')
        archive.unlink()
    embedding = folder / '3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx'
    if not embedding.is_file():
        pending = embedding.with_suffix('.download')
        urlretrieve(BASE + 'speaker-recongition-models/' + embedding.name, pending)
        pending.replace(embedding)
    print('Local diarization models are ready.')


if __name__ == '__main__':
    install_models()
