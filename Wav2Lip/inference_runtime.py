"""OpenVINO adapters for the existing Wav2Lip and SFD weights."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def compiled_model(model, example, cache_name, checkpoint, device, dynamic=False):
    import openvino as ov
    cache = ROOT / 'jobs' / 'models' / 'openvino'
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    identity = f'{checkpoint.resolve()}:{checkpoint.stat().st_size}:{checkpoint.stat().st_mtime_ns}'
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
    path = cache / f'{cache_name}_nb_{suffix}.xml'
    if not path.exists():
        # Dynamic batch lets one inference call cover a whole frame batch on CPU.
        shapes = [(-1,) + tuple(t.shape[1:]) for t in example]
        if dynamic:
            shapes = [(1, 3, -1, -1)]
        converted = ov.convert_model(model, example_input=example, input=shapes)
        ov.save_model(converted, path, compress_to_fp16=False)
    core = ov.Core()
    device = effective_device(core, device)
    compiled = core.compile_model(path, device, {'PERFORMANCE_HINT': 'LATENCY',
                                               'CACHE_DIR': str(cache / 'compiled')})
    return compiled, device


def effective_device(core, device):
    """Use the requested OpenVINO device, or CPU when it is not present."""
    if device in core.available_devices:
        return device
    if 'CPU' not in core.available_devices:
        raise RuntimeError(f'OpenVINO device {device} is unavailable; available: {core.available_devices}')
    print(f'OpenVINO device {device} is unavailable; using CPU', flush=True)
    return 'CPU'


class MouthInference:
    def __init__(self, model, checkpoint, device):
        self.compiled, self.device = compiled_model(model, (torch.zeros(1, 1, 80, 16), torch.zeros(1, 6, 96, 96)),
                                                    'wav2lip', checkpoint, device)

    def __call__(self, mel, images):
        result = self.compiled([mel.cpu().numpy(), images.cpu().numpy()])[0]
        return torch.from_numpy(np.asarray(result))


class FaceInference:
    def __init__(self, model, checkpoint, device):
        self.compiled, self.device = compiled_model(model, (torch.zeros(1, 3, 270, 480),),
                                                    'sfd', checkpoint, device, dynamic=True)

    def __call__(self, images):
        batches = [self.compiled([image[None]]) for image in images.cpu().numpy()]
        return [torch.from_numpy(np.concatenate([batch[index] for batch in batches]))
                for index in range(len(self.compiled.outputs))]
