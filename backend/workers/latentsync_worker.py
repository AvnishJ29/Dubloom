"""LatentSync 1.6 lip sync of the speaking face only.

Runs inside .venv-latentsync. Instead of LatentSync's own "largest face in
every frame" detection (which fails on frames without a face and cannot tell
speakers apart), it takes per-frame targets from the studio: the box and 106
landmarks of the face that speaks in each frame. Contiguous runs of the same
face are rendered as spans; every other frame is copied unchanged.

With --cache, each rendered span is kept under a key of everything it depends
on (picture, landmarks, whisper features, settings), so after a script or
voice edit only spans in the 30-second whisper windows (LatentSync encodes
audio in fixed 30 s windows) around the changed speech are diffused again.

    python latentsync_worker.py VIDEO.mp4 VOICE.wav TARGETS.npz OUT.mp4 --repo third_party/LatentSync \
        --checkpoints models/latentsync-1.6 --vae models/sd-vae-ft-mse [--steps 20] [--guidance 1.5] [--deepcache]

torch, diffusers and LatentSync come from .venv-latentsync and --repo, so they are imported
inside the functions that use them, once `enter_latentsync` has put the repo on the path.
"""

import argparse
import gc
import glob
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import cv2
from paste import paste_generated_region
import numpy as np

# frames per diffusion window (LatentSync's num_frames); --batch can lower it for CPU tests
BATCH = 16
MAX_SPAN = 400  # long spans are rendered in pieces to bound memory
RESOLUTION = 512
AUDIO_MARGIN = 1.0  # seconds of audio around a span that its Whisper features may depend on
FEATURE_FPS = 25  # Whisper chunks per second, one per frame of the 25 fps render source


# ---- command line and environment ----


def parse_args():
    """The worker's command line (see the module docstring)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("audio")
    parser.add_argument("targets")
    parser.add_argument("output")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--deepcache", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1247)
    # x264 quality of the full-video output, capped at 8 (near-lossless): the picture is encoded again later
    parser.add_argument("--crf", type=int, default=8)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument(
        "--cache", help="directory for rendered spans, reused when nothing they depend on changed"
    )
    parser.add_argument("--video-key", default="", help="digest of VIDEO (part of every span key)")
    parser.add_argument(
        "--spans-only",
        action="store_true",
        help="render and cache the spans only, write spans.json; no full video",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=1,
        help="16-frame windows denoised together in one UNet batch (2 fills the H100 better)",
    )
    return parser.parse_args()


def enter_latentsync(repo):
    """Work from the LatentSync checkout (its configs are relative paths and it is not installed)
    and return its 512 px UNet config.

    Everything the render needs is imported here, on the main thread, so a broken install fails
    before any work and the span threads (`render_spans_only`) never import a module for the
    first time concurrently.
    """
    os.chdir(repo)
    sys.path.insert(0, os.getcwd())
    import torch

    # inference only: never keep autograd state (large on 512 px latents)
    torch.set_grad_enabled(False)
    for module in (
        "torchvision",
        "diffusers",
        "accelerate.utils",
        "latentsync.models.unet",
        "latentsync.pipelines.lipsync_pipeline",
        "latentsync.whisper.audio2feature",
        "latentsync.utils.affine_transform",
        "latentsync.utils.image_processor",
    ):
        importlib.import_module(module)
    from omegaconf import OmegaConf

    return OmegaConf.load("configs/unet/stage2_512.yaml")


def choose_device(name, dtype_name):
    """The torch device and dtype to render with (fp16 on GPUs newer than Turing for "auto")."""
    import torch

    # a wrong GPU slot must not become a silent CPU render
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"cuda requested but no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    use_cuda = name == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    if dtype_name == "auto":
        dtype = (
            torch.float16
            if use_cuda and torch.cuda.get_device_capability()[0] > 7
            else torch.float32
        )
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    return device, dtype


def load_whisper(checkpoints, config, device):
    """LatentSync's Whisper tiny audio encoder."""
    from latentsync.whisper.audio2feature import Audio2Feature

    return Audio2Feature(
        model_path=os.path.join(checkpoints, "whisper", "tiny.pt"),
        device=device,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )


# ---- span planning and audio features ----


def spans_of(track: np.ndarray):
    """[start, end) frame spans of one target face each, at most MAX_SPAN frames long.

    `track` holds the target track id per frame; frames with -1 belong to no span.
    """
    spans, start = [], None
    for index, value in enumerate(track.tolist() + [-1]):
        if start is not None and (value != track[start] or index - start >= MAX_SPAN):
            spans.append((start, index))
            start = None
        if start is None and value >= 0:
            start = index
    return spans


def chunk_indices(chunks, base, first, count, window):
    """Whisper chunk of every frame in a window.

    The last window is padded by holding its final frame; frames past the last chunk reuse it.
    """
    return [max(0, min(first - base + min(k, count - 1), len(chunks) - 1)) for k in range(window)]


class SpanAudio:
    """Whisper features per span, from the span's own audio (plus a margin), not the whole track.

    Whisper encodes 30 s windows with global attention, so features of a frame used to change
    whenever any line in the same 30 s changed: one edited line re-rendered most of the video.
    Per-span audio keeps a span's key and output dependent on its own lines only.
    """

    def __init__(self, path, encoder):
        self.path, self.encoder = path, encoder
        with wave.open(path, "rb") as reader:
            self.rate = reader.getframerate()
            self.samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
            if reader.getnchannels() > 1:
                self.samples = (
                    self.samples.reshape(-1, reader.getnchannels()).mean(axis=1).astype("<i2")
                )
        self._whole = None
        self._spans = {}

    def whole_chunks(self):
        """Chunks of the whole track (the fallback), computed once."""
        if self._whole is None:
            self._whole = self._chunks(self.path)
        return self._whole

    def features(self, span):
        """(chunks, base frame) for a span: chunk of frame f is chunks[f - base]."""
        if span not in self._spans:
            self._spans[span] = self._span_features(*span)
        return self._spans[span]

    def _chunks(self, path):
        return self.encoder.feature2chunks(
            feature_array=self.encoder.audio2feat(path), fps=FEATURE_FPS
        )

    def _span_features(self, first, end):
        t0, t1 = max(0.0, first / FEATURE_FPS - AUDIO_MARGIN), end / FEATURE_FPS + AUDIO_MARGIN
        piece = self.samples[int(t0 * self.rate) : int(t1 * self.rate)]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                with wave.open(handle, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(self.rate)
                    writer.writeframes(piece.tobytes())
                chunks = self._chunks(handle.name)
            os.unlink(handle.name)
            if len(chunks) < end - first:
                raise ValueError("short feature run")
            return chunks, int(round(t0 * FEATURE_FPS))
        except Exception as exc:  # noqa: BLE001
            # the slice gave too few feature chunks (empty or cut short): the whole-track features still work
            print(f"span {first}-{end}: whole-track audio features ({exc})", flush=True)
            return self.whole_chunks(), 0


# ---- span cache ----


def span_signature(args, checkpoints, dtype, window):
    """Settings part of every span key: a span's output depends only on these inputs, its pixels
    and its audio (the noise is seeded and shared)."""
    return "|".join(
        map(
            str,
            [
                "span-audio-v2",
                args.video_key,
                os.path.getsize(os.path.join(checkpoints, "latentsync_unet.pt")),
                args.steps,
                args.guidance,
                args.deepcache,
                args.seed,
                dtype,
                window,
                RESOLUTION,
            ],
        )
    )


def span_key(signature, lmk, track, audio, span, window):
    """Cache key of a span: settings, landmarks, track and the Whisper features it uses."""
    first, end = span
    digest = hashlib.sha256(signature.encode())
    digest.update(np.ascontiguousarray(lmk[first:end]).tobytes())
    digest.update(str(int(track[first])).encode())
    chunks, base = audio.features(span)
    for start in range(first, end, window):
        for index in chunk_indices(chunks, base, start, min(window, end - start), window):
            digest.update(chunks[index].detach().cpu().numpy().astype(np.float16).tobytes())
    return digest.hexdigest()[:32]


def span_path(cache_dir, key):
    """Cache file of the span with this key."""
    return os.path.join(cache_dir, f"span_{key}.mp4")


def drop_stale_spans(cache_dir, keys):
    """Delete cached spans no longer in the timeline, and broken writes."""
    wanted = {f"span_{key}.mp4" for key in keys.values()}
    for stale in glob.glob(os.path.join(cache_dir, "span_*")):
        # span_<key>.<restore stamp>.mp4 stays with its span
        if os.path.basename(stale).split(".")[0] + ".mp4" not in wanted:
            os.remove(stale)


def read_cached_span(path, length):
    """Frames of a cached span, or None when absent or incomplete."""
    if not os.path.isfile(path):
        return None
    reader, frames = cv2.VideoCapture(path), []
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        frames.append(frame)
    reader.release()
    return frames if len(frames) == length else None


def store_span(path, frames, fps):
    """Keep a rendered span (near-lossless) for later runs; written atomically."""
    height, width = frames[0].shape[:2]
    partial = path + ".part"
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "mp4",
            partial,
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() == 0:
        os.replace(partial, path)
    elif os.path.exists(partial):
        os.remove(partial)


# ---- rendering one span: align (CPU), diffuse (GPU), paste back (CPU) ----


class Aligned(NamedTuple):
    """The 512 px face crops of a span and what is needed to paste them back."""

    faces: object  # (frames, 3, 512, 512) uint8 tensor
    matrices: list
    sizes: list
    restorer: object  # the AlignRestore that warped them


def _split_windows(latent, n, window, do_cfg):
    """(1, C, n*window, h, w) [or 2x that for CFG] -> (n, C, window, h, w) [or 2n]: one batch element per window."""
    import torch

    halves = latent.chunk(2) if do_cfg else (latent,)
    out = []
    for half in halves:
        out.append(torch.cat([half[:, :, k * window : (k + 1) * window] for k in range(n)], dim=0))
    return torch.cat(out) if do_cfg else out[0]


class SpanRenderer:
    """LatentSync on the target face of one span at a time."""

    def __init__(self, args, config, checkpoints, vae_dir, audio, lmk, window, device, dtype):
        from torchvision import transforms

        self.args, self.config = args, config
        self.checkpoints, self.vae_dir = checkpoints, vae_dir
        self.audio, self.lmk, self.window = audio, lmk, window
        self.device, self.dtype = device, dtype
        self.do_cfg = args.guidance > 1.0
        self.normalize = transforms.Normalize([0.5], [0.5])
        self._models = {}

    def load_models(self):
        """The diffusion models are loaded only if some span actually needs rendering."""
        if self._models:
            return self._models
        from omegaconf import OmegaConf
        from diffusers import AutoencoderKL, DDIMScheduler
        from accelerate.utils import set_seed
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from latentsync.utils.image_processor import load_fixed_mask

        args, dtype, device = self.args, self.dtype, self.device
        scheduler = DDIMScheduler.from_pretrained("configs")
        vae = AutoencoderKL.from_pretrained(self.vae_dir, torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            os.path.join(self.checkpoints, "latentsync_unet.pt"),
            device="cpu",
        )
        unet = unet.to(dtype=dtype)
        gc.collect()
        pipeline = LipsyncPipeline(
            vae=vae, audio_encoder=self.audio.encoder, unet=unet, scheduler=scheduler
        ).to(device)
        if args.deepcache:
            from DeepCache import DeepCacheSDHelper

            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        set_seed(args.seed)
        self._models.update(
            pipeline=pipeline,
            scheduler=scheduler,
            unet=unet,
            mask=load_fixed_mask(RESOLUTION, self.config.data.mask_image_path),
            base_latents=pipeline.prepare_latents(
                self.window, vae.config.latent_channels, RESOLUTION, RESOLUTION, dtype, device, None
            ),
            extra=pipeline.prepare_extra_step_kwargs(None, 0.0),
        )
        return self._models

    def render(self, frames_rgb, first):
        """The lip-synced frames of the span starting at frame `first`."""
        aligned = self.align_faces(frames_rgb, first)
        generated = self.diffuse(aligned.faces, first)
        return self.paste_back(frames_rgb, generated, aligned)

    def align_faces(self, frames_rgb, first):
        """CPU: the smoothed affine crop of every frame of a span (fresh smoothing per span)."""
        import torch
        from latentsync.utils.affine_transform import AlignRestore

        restorer = AlignRestore(
            resolution=RESOLUTION,
            device=self.device,
            dtype=self.dtype if self.device.type == "cuda" else torch.float32,
        )
        faces, matrices, sizes = [], [], []
        for offset, frame in enumerate(frames_rgb):
            points = self.lmk[first + offset]
            landmarks3 = np.round(
                [
                    np.mean(points[[43, 48, 49, 51, 50]], axis=0),
                    np.mean(points[101:106], axis=0),
                    np.mean(points[[74, 77, 83, 86]], axis=0),
                ]
            )
            face, matrix = restorer.align_warp_face(
                frame.copy(), landmarks3=landmarks3, smooth=True
            )
            sizes.append((face.shape[0], face.shape[1]))
            face = cv2.resize(face, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LANCZOS4)
            faces.append(torch.from_numpy(face).permute(2, 0, 1))
            matrices.append(matrix)
        return Aligned(torch.stack(faces), matrices, sizes, restorer)

    def diffuse(self, faces, first):
        """GPU: the lip-synced 512 px faces of a span, a group of 16-frame windows at a time."""
        import torch

        models = self.load_models()
        chunks, base = self.audio.features((first, first + len(faces)))
        # windows denoised together: same noise per window, so the output is unchanged
        group = max(1, self.args.windows)
        starts = list(range(0, len(faces), self.window))
        outputs = []
        for g in range(0, len(starts), group):
            batches, reals, embeds = self._window_inputs(
                faces, starts[g : g + group], chunks, base, first
            )
            decoded = self._denoise(models, batches, embeds)
            for k, real in enumerate(reals):
                outputs.append(decoded[k * self.window : k * self.window + real])
        return torch.cat(outputs)

    def _window_inputs(self, faces, starts, chunks, base, first):
        """Faces, real frame counts and Whisper chunks of the windows starting at `starts`."""
        import torch

        window = self.window
        batches, reals, embeds = [], [], []
        for start in starts:
            batch = faces[start : start + window]
            real = len(batch)
            if real < window:  # pad the last window by holding its final frame
                batch = torch.cat([batch, batch[-1:].repeat(window - real, 1, 1, 1)])
            indices = chunk_indices(chunks, base, first + start, real, window)
            batches.append(batch)
            reals.append(real)
            embeds.append(
                torch.stack([chunks[i] for i in indices]).to(self.device, dtype=self.dtype)
            )
        return batches, reals, embeds

    def _denoise(self, loaded, batches, embeds):
        """The DDIM loop on n windows in one UNet batch; the (n*window) decoded faces."""
        import torch

        pipeline, scheduler, unet, mask = (
            loaded["pipeline"],
            loaded["scheduler"],
            loaded["unet"],
            loaded["mask"],
        )
        base_latents, extra_step_kwargs = loaded["base_latents"], loaded["extra"]
        window, device, dtype, do_cfg = self.window, self.device, self.dtype, self.do_cfg
        n = len(batches)
        pixel_values = self.normalize(torch.cat(batches).float() / 255.0)  # (n*window, 3, H, W)
        masked_pixel_values = pixel_values * mask
        masks = mask[0:1].unsqueeze(0).repeat(n * window, 1, 1, 1)
        # (n, window, ...) or (window, ...)
        audio_embeds = torch.stack(embeds) if n > 1 else embeds[0]
        if do_cfg:
            audio_embeds = torch.cat([torch.zeros_like(audio_embeds), audio_embeds])
        mask_latents, masked_image_latents = pipeline.prepare_mask_latents(
            masks, masked_pixel_values, RESOLUTION, RESOLUTION, dtype, device, None, do_cfg
        )
        ref_latents = pipeline.prepare_image_latents(pixel_values, device, dtype, None, do_cfg)
        if n > 1:  # (1, C, n*window, h, w) -> (n, C, window, h, w): one sample per window
            mask_latents = _split_windows(mask_latents, n, window, do_cfg)
            masked_image_latents = _split_windows(masked_image_latents, n, window, do_cfg)
            ref_latents = _split_windows(ref_latents, n, window, do_cfg)
        latents = base_latents.clone().repeat(n, 1, 1, 1, 1) if n > 1 else base_latents.clone()
        scheduler.set_timesteps(self.args.steps, device=device)
        with torch.no_grad():
            for t in scheduler.timesteps:
                unet_input = torch.cat([latents] * 2) if do_cfg else latents
                unet_input = scheduler.scale_model_input(unet_input, t)
                unet_input = torch.cat(
                    [unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                )
                noise = unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                if do_cfg:
                    uncond, conditioned = noise.chunk(2)
                    noise = uncond + self.args.guidance * (conditioned - uncond)
                latents = scheduler.step(noise, t, latents, **extra_step_kwargs).prev_sample
            if n > 1:  # back to (1, C, n*window, h, w)
                latents = torch.cat([latents[k] for k in range(n)], dim=1).unsqueeze(0)
            decoded = pipeline.decode_latents(latents)
            return pipeline.paste_surrounding_pixels_back(
                decoded, pixel_values, 1 - masks, device, dtype
            )

    def paste_back(self, frames_rgb, generated, aligned):
        """CPU: the generated lower face back into every original frame."""
        import torchvision
        from torchvision import transforms

        loaded = self.load_models()
        # the inpainted lower face, 512 x 512
        region = ((1 - loaded["mask"][0].float()).cpu().numpy() * 255).astype(np.uint8)
        restored = []
        for offset, frame in enumerate(frames_rgb):
            height, width = aligned.sizes[offset]
            face = torchvision.transforms.functional.resize(
                generated[offset].float(),
                size=(height, width),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            pasted = aligned.restorer.restore_img(frame, face, aligned.matrices[offset])
            pasted = pasted.cpu().numpy() if hasattr(pasted, "cpu") else np.asarray(pasted)
            frame_to_crop = aligned.matrices[offset].detach().float().cpu().numpy().reshape(2, 3)
            restored.append(
                paste_generated_region(frame, pasted, region, frame_to_crop, width, height)
            )
        return restored


# ---- outputs: cached spans + manifest (--spans-only), or the whole video ----


def read_frames(capture, count):
    """Up to `count` next frames of `capture` (fewer at the end of the video)."""
    frames = []
    for _ in range(count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def decode_span(video, span):
    """The RGB frames of a span of VIDEO."""
    first, end = span
    reader = cv2.VideoCapture(video)
    reader.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in read_frames(reader, end - first)]
    reader.release()
    if len(frames) != end - first:
        raise RuntimeError(f"could not read frames {first}-{end} of {video}")
    return frames


def write_manifest(cache_dir, manifest, total_frames, size, fps, complete):
    """spans.json, rewritten as spans land, so a preview can be built from what exists so far."""
    width, height = size
    payload = {
        "frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
        "complete": complete,
        "spans": manifest,
    }
    partial = os.path.join(cache_dir, "spans.json.part")
    with open(partial, "w") as stream:
        json.dump(payload, stream)
    os.replace(partial, os.path.join(cache_dir, "spans.json"))


def render_spans_only(renderer, video, spans, keys, cache_dir, total_frames, size, fps):
    """Render every span missing from the cache into it, with spans.json listing all spans.

    The GPU diffuses one span while a worker thread decodes and aligns the next one and another
    pastes back and stores the previous one.
    """
    total_target = int(sum(b - a for a, b in spans))
    manifest, started, done, reused = [], time.time(), 0, 0
    todo = []
    for first, end in spans:
        span = (first, end)
        path = span_path(cache_dir, keys[span])
        manifest.append({"first": first, "end": end, "key": keys[span], "file": path})
        if os.path.isfile(path) and read_cached_span(path, end - first) is not None:
            reused += end - first
        else:
            todo.append((span, path))
    write_manifest(cache_dir, manifest, total_frames, size, fps, False)

    def prepare(span):
        """CPU, ahead of the GPU: decode the span's frames and align the faces."""
        frames = decode_span(video, span)
        return frames, renderer.align_faces(frames, span[0])

    def finish(span, path, frames, generated, aligned):
        """CPU, behind the GPU: paste the faces back and store the span."""
        rendered = renderer.paste_back(frames, generated, aligned)
        store_span(path, [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rendered], fps)
        return span[1] - span[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        ahead = pool.submit(prepare, todo[0][0]) if todo else None
        pending = []
        for index, (span, path) in enumerate(todo):
            frames, aligned = ahead.result()
            ahead = pool.submit(prepare, todo[index + 1][0]) if index + 1 < len(todo) else None
            # the GPU part; the next span aligns meanwhile
            generated = renderer.diffuse(aligned.faces, span[0])
            pending.append(pool.submit(finish, span, path, frames, generated, aligned))
            for future in [f for f in pending if f.done()]:
                done += future.result()
                pending.remove(future)
            write_manifest(cache_dir, manifest, total_frames, size, fps, False)
            print(
                f"Rendered {done} of {total_target} frames ({time.time() - started:.0f}s)",
                flush=True,
            )
        for future in pending:
            done += future.result()
    write_manifest(cache_dir, manifest, total_frames, size, fps, True)
    print(
        f"DONE frames={total_frames} synced={done + reused} rendered={done} reused={reused} seconds={time.time() - started:.1f}",
        flush=True,
    )
    return 0


def copy_until(capture, writer, position, stop):
    """Copy frames unchanged from `position` up to `stop`; the new position (short at the end)."""
    while position < stop:
        ok, frame = capture.read()
        if not ok:
            break
        writer.stdin.write(frame.tobytes())
        position += 1
    return position


def render_video(renderer, capture, output, spans, keys, cache_dir, size, fps, crf):
    """The whole video to OUTPUT: spans lip-synced (or from the cache), other frames copied."""
    width, height = size
    total_target = int(sum(b - a for a, b in spans))
    writer = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "faster",
            "-crf",
            str(min(crf, 8)),
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        stdin=subprocess.PIPE,
    )
    position, done, reused, started = 0, 0, 0, time.time()
    for first, end in spans:
        position = copy_until(capture, writer, position, first)
        if position < first:
            break
        frames = read_frames(capture, end - first)
        if not frames:
            break
        span = (first, end)
        complete = len(frames) == end - first
        output_frames = (
            read_cached_span(span_path(cache_dir, keys[span]), end - first)
            if cache_dir and complete
            else None
        )
        if output_frames is not None:
            reused += len(frames)
        else:
            rendered = renderer.render([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], first)
            output_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rendered]
            if cache_dir and complete:
                store_span(span_path(cache_dir, keys[span]), output_frames, fps)
        for frame in output_frames:
            writer.stdin.write(frame.tobytes())
        position += len(frames)
        done += len(frames)
        print(
            f"Rendered {done} of {total_target} frames ({time.time() - started:.0f}s)", flush=True
        )
    position = copy_until(capture, writer, position, 10**9)
    capture.release()
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg could not write the lip-synced video")
    print(
        f"DONE frames={position} synced={done} reused={reused} seconds={time.time() - started:.1f}",
        flush=True,
    )


def main():
    """Lip-sync the target spans of VIDEO to VOICE.

    Writes the full video to OUT.mp4 (untargeted frames copied), or with --spans-only only the
    rendered spans and spans.json in the --cache directory. Prints "Rendered n of m frames"
    progress and a final DONE line.
    """
    args = parse_args()
    window = max(1, args.batch)
    video, audio_path, targets_path, output = (
        os.path.abspath(p) for p in (args.video, args.audio, args.targets, args.output)
    )
    checkpoints, vae_dir = os.path.abspath(args.checkpoints), os.path.abspath(args.vae)
    config = enter_latentsync(args.repo)
    device, dtype = choose_device(args.device, args.dtype)
    encoder = load_whisper(checkpoints, config, device)
    print(f"device {device} dtype {dtype}", flush=True)

    targets = np.load(targets_path)
    lmk, track = targets["lmk"], targets["track"]
    spans = spans_of(track)
    audio = SpanAudio(audio_path, encoder)
    signature = span_signature(args, checkpoints, dtype, window)
    # resolved after the chdir, so a relative --cache lies under --repo
    cache_dir = os.path.abspath(args.cache) if args.cache else None
    keys = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        keys = {span: span_key(signature, lmk, track, audio, span, window) for span in spans}
        drop_stale_spans(cache_dir, keys)

    renderer = SpanRenderer(args, config, checkpoints, vae_dir, audio, lmk, window, device, dtype)
    capture = cv2.VideoCapture(video)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    if args.spans_only:
        # the finisher (backend/rendering/finish.py) pastes the cached spans into the picture and
        # encodes once, so nothing is written here but the spans and a manifest; cached spans are
        # not even decoded
        if not cache_dir:
            raise SystemExit("--spans-only needs --cache")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return render_spans_only(renderer, video, spans, keys, cache_dir, total_frames, size, fps)
    return render_video(renderer, capture, output, spans, keys, cache_dir, size, fps, args.crf)


if __name__ == "__main__":
    sys.exit(main())
