"""Mouth-region super-resolution with the compact Real-ESRGAN (realesr-general-x4v3).

Wav2Lip predicts a 96x96 mouth. Up-sampling it 4x with a learned model before
blending gives visibly sharper lips and teeth than bicubic interpolation when
the face on screen is larger than the model's native resolution.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """Architecture of realesr-general-x4v3 (Real-ESRGAN, compact VGG-style SR)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


class MouthEnhancer:
    def __init__(self, weights: Path, runtime: str = 'torch', openvino_device: str = 'CPU', batch: int = 8):
        model = SRVGGNetCompact()
        state = torch.load(str(weights), map_location='cpu', weights_only=False)
        state = state.get('params_ema') or state.get('params') or state
        model.load_state_dict(state, strict=True)
        model.eval()
        self.batch = batch
        self.compiled = None
        if runtime == 'openvino':
            from inference_runtime import compiled_model
            self.compiled, self.device = compiled_model(model, (torch.zeros(1, 3, 96, 96),), 'esrgan',
                                                        weights, openvino_device)
        self.model = model

    def __call__(self, faces: np.ndarray) -> np.ndarray:
        """faces: (N, 96, 96, 3) BGR uint8 -> (N, 384, 384, 3) BGR uint8."""
        if not len(faces):
            return faces
        rgb = np.ascontiguousarray(faces[..., ::-1]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb.transpose(0, 3, 1, 2))
        outputs = []
        for start in range(0, len(tensor), self.batch):
            chunk = tensor[start:start + self.batch]
            if self.compiled is not None:
                result = torch.from_numpy(np.asarray(self.compiled([chunk.numpy()])[0]))
            else:
                with torch.inference_mode():
                    result = self.model(chunk)
            outputs.append(result)
        merged = torch.cat(outputs).clamp_(0, 1).permute(0, 2, 3, 1).numpy() * 255
        return np.ascontiguousarray(merged.round().astype(np.uint8)[..., ::-1])
