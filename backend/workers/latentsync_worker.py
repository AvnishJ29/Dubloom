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
"""

import argparse
import glob
import json
import hashlib
import os
import subprocess
import sys
import time
import wave
import tempfile

import cv2
from paste import paste_generated_region
import numpy as np

BATCH = (
    16  # frames per diffusion window (LatentSync's num_frames); --batch can lower it for CPU tests
)
MAX_SPAN = 400  # long spans are rendered in pieces to bound memory
RESOLUTION = 512
AUDIO_MARGIN = 1.0  # seconds of audio around a span that its Whisper features may depend on


def _split_windows(latent, n, window, do_cfg):
    """(1, C, n*window, h, w) [or 2x that for CFG] -> (n, C, window, h, w) [or 2n]: one batch element per window."""
    import torch

    halves = latent.chunk(2) if do_cfg else (latent,)
    out = []
    for half in halves:
        out.append(torch.cat([half[:, :, k * window : (k + 1) * window] for k in range(n)], dim=0))
    return torch.cat(out) if do_cfg else out[0]


def spans_of(track: np.ndarray):
    spans, start = [], None
    for index, value in enumerate(track.tolist() + [-1]):
        if start is not None and (value != track[start] or index - start >= MAX_SPAN):
            spans.append((start, index))
            start = None
        if start is None and value >= 0:
            start = index
    return spans


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


def main():
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
    parser.add_argument("--crf", type=int, default=16)
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
    args = parser.parse_args()
    window = max(1, args.batch)
    video, audio, targets_path, output = (
        os.path.abspath(p) for p in (args.video, args.audio, args.targets, args.output)
    )
    checkpoints, vae_dir = os.path.abspath(args.checkpoints), os.path.abspath(args.vae)
    os.chdir(args.repo)
    sys.path.insert(0, os.getcwd())

    import torch
    import torchvision

    torch.set_grad_enabled(
        False
    )  # inference only: never keep autograd state (large on 512 px latents)
    from torchvision import transforms
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from accelerate.utils import set_seed
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from latentsync.utils.affine_transform import AlignRestore
    from latentsync.utils.image_processor import load_fixed_mask

    if (
        args.device == "cuda" and not torch.cuda.is_available()
    ):  # a wrong GPU slot must not become a silent CPU render
        raise SystemExit(
            f"cuda requested but no GPU is visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    use_cuda = args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.dtype == "auto":
        dtype = (
            torch.float16
            if use_cuda and torch.cuda.get_device_capability()[0] > 7
            else torch.float32
        )
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    config = OmegaConf.load("configs/unet/stage2_512.yaml")
    audio_encoder = Audio2Feature(
        model_path=os.path.join(checkpoints, "whisper", "tiny.pt"),
        device=device,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    print(f"device {device} dtype {dtype}", flush=True)

    data = np.load(targets_path)
    lmk, track = data["lmk"], data["track"]
    spans = spans_of(track)
    total_target = int(sum(b - a for a, b in spans))
    do_cfg = args.guidance > 1.0

    # ---- Whisper features per span, from the span's own audio (plus a margin), not from the whole track ----
    # Whisper encodes 30 s windows with global attention, so features of a frame used to change whenever any line in the
    # same 30 s changed: one edited line re-rendered most of the video. Per-span audio keeps a span's key and output
    # dependent on its own lines only.
    with wave.open(audio, "rb") as reader:
        rate = reader.getframerate()
        samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        if reader.getnchannels() > 1:
            samples = samples.reshape(-1, reader.getnchannels()).mean(axis=1).astype("<i2")
    whole = {}

    def whole_chunks():
        if "chunks" not in whole:
            whole["chunks"] = audio_encoder.feature2chunks(
                feature_array=audio_encoder.audio2feat(audio), fps=25
            )
        return whole["chunks"]

    audio_for = {}

    def span_audio(span):
        """(chunks, base frame) for a span: chunk of frame f is chunks[f - base]."""
        if span in audio_for:
            return audio_for[span]
        first, end = span
        t0, t1 = max(0.0, first / 25 - AUDIO_MARGIN), end / 25 + AUDIO_MARGIN
        piece = samples[int(t0 * rate) : int(t1 * rate)]
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                with wave.open(handle, "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                    writer.writeframes(piece.tobytes())
                chunks = audio_encoder.feature2chunks(
                    feature_array=audio_encoder.audio2feat(handle.name), fps=25
                )
            os.unlink(handle.name)
            if len(chunks) < end - first:
                raise ValueError("short feature run")
            audio_for[span] = (chunks, int(round(t0 * 25)))
        except (
            Exception
        ) as exc:  # noqa: BLE001  (no speech decoded in the slice: the whole-track features still work)
            print(f"span {first}-{end}: whole-track audio features ({exc})", flush=True)
            audio_for[span] = (whole_chunks(), 0)
        return audio_for[span]

    def chunk_indices(chunks, base, first, count):
        """Whisper chunk of every frame in a window (the last window is padded by holding its final frame)."""
        return [
            max(0, min(first - base + min(k, count - 1), len(chunks) - 1)) for k in range(window)
        ]

    # ---- span cache: a span's output depends only on these inputs (the noise is seeded and shared) ----
    signature = "|".join(
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

    def span_key(first, end):
        digest = hashlib.sha256(signature.encode())
        digest.update(np.ascontiguousarray(lmk[first:end]).tobytes())
        digest.update(str(int(track[first])).encode())
        chunks, base = span_audio((first, end))
        for start in range(first, end, window):
            for index in chunk_indices(chunks, base, start, min(window, end - start)):
                digest.update(chunks[index].detach().cpu().numpy().astype(np.float16).tobytes())
        return digest.hexdigest()[:32]

    cache_dir = os.path.abspath(args.cache) if args.cache else None
    keys = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        keys = {span: span_key(*span) for span in spans}
        wanted = {f"span_{key}.mp4" for key in keys.values()}
        for stale in glob.glob(
            os.path.join(cache_dir, "span_*")
        ):  # spans no longer in the timeline, broken writes
            if (
                os.path.basename(stale).split(".")[0] + ".mp4" not in wanted
            ):  # span_<key>.<restore stamp>.mp4 stays with its span
                os.remove(stale)

    def cached_frames(span):
        path = os.path.join(cache_dir, f"span_{keys[span]}.mp4") if cache_dir else None
        if not path or not os.path.isfile(path):
            return None
        reader, frames = cv2.VideoCapture(path), []
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
        reader.release()
        return frames if len(frames) == span[1] - span[0] else None

    models = {}

    def load_models():
        """The diffusion models are loaded only if some span actually needs rendering."""
        if models:
            return models
        scheduler = DDIMScheduler.from_pretrained("configs")
        vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            os.path.join(checkpoints, "latentsync_unet.pt"),
            device="cpu",
        )
        unet = unet.to(dtype=dtype)
        import gc

        gc.collect()
        pipeline = LipsyncPipeline(
            vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler
        ).to(device)
        if args.deepcache:
            from DeepCache import DeepCacheSDHelper

            helper = DeepCacheSDHelper(pipe=pipeline)
            helper.set_params(cache_interval=3, cache_branch_id=0)
            helper.enable()
        set_seed(args.seed)
        models.update(
            pipeline=pipeline,
            scheduler=scheduler,
            unet=unet,
            mask=load_fixed_mask(RESOLUTION, config.data.mask_image_path),
            base_latents=pipeline.prepare_latents(
                window, vae.config.latent_channels, RESOLUTION, RESOLUTION, dtype, device, None
            ),
            extra=pipeline.prepare_extra_step_kwargs(None, 0.0),
        )
        return models

    normalize = transforms.Normalize([0.5], [0.5])

    def align_faces(frames_rgb, first):
        """CPU: the smoothed affine crop of every frame of a span (fresh smoothing per span)."""
        restorer = AlignRestore(
            resolution=RESOLUTION, device=device, dtype=dtype if use_cuda else torch.float32
        )
        faces, matrices, sizes = [], [], []
        for offset, frame in enumerate(frames_rgb):
            points = lmk[first + offset]
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
        return torch.stack(faces), matrices, sizes, restorer

    def paste_back(frames_rgb, generated, matrices, sizes, restorer):
        """CPU: the generated lower face back into every original frame."""
        loaded = load_models()
        region = ((1 - loaded["mask"][0].float()).cpu().numpy() * 255).astype(
            np.uint8
        )  # the inpainted lower face, 512 x 512
        restored = []
        for offset, frame in enumerate(frames_rgb):
            height, width = sizes[offset]
            face = torchvision.transforms.functional.resize(
                generated[offset].float(),
                size=(height, width),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
            pasted = restorer.restore_img(frame, face, matrices[offset])
            pasted = pasted.cpu().numpy() if hasattr(pasted, "cpu") else np.asarray(pasted)
            frame_to_crop = matrices[offset].detach().float().cpu().numpy().reshape(2, 3)
            restored.append(
                paste_generated_region(frame, pasted, region, frame_to_crop, width, height)
            )
        return restored

    def render(frames_rgb, first):
        faces, matrices, sizes, restorer = align_faces(frames_rgb, first)
        generated = diffuse(faces, first)
        return paste_back(frames_rgb, generated, matrices, sizes, restorer)

    def diffuse(faces, first):
        """GPU: the lip-synced 512 px faces of a span."""
        loaded = load_models()
        chunks, base = span_audio((first, first + len(faces)))
        pipeline, scheduler, unet, mask = (
            loaded["pipeline"],
            loaded["scheduler"],
            loaded["unet"],
            loaded["mask"],
        )
        base_latents, extra_step_kwargs = loaded["base_latents"], loaded["extra"]
        outputs = []
        group = max(
            1, args.windows
        )  # windows denoised together: same noise per window, so the output is unchanged
        starts = list(range(0, len(faces), window))
        for g in range(0, len(starts), group):
            members = starts[g : g + group]
            batches, reals, embeds = [], [], []
            for start in members:
                batch = faces[start : start + window]
                real = len(batch)
                if real < window:  # pad the last window by holding its final frame
                    batch = torch.cat([batch, batch[-1:].repeat(window - real, 1, 1, 1)])
                indices = chunk_indices(chunks, base, first + start, real)
                batches.append(batch)
                reals.append(real)
                embeds.append(torch.stack([chunks[i] for i in indices]).to(device, dtype=dtype))
            n = len(members)
            pixel_values = normalize(torch.cat(batches).float() / 255.0)  # (n*window, 3, H, W)
            masked_pixel_values = pixel_values * mask
            masks = mask[0:1].unsqueeze(0).repeat(n * window, 1, 1, 1)
            audio_embeds = (
                torch.stack(embeds) if n > 1 else embeds[0]
            )  # (n, window, ...) or (window, ...)
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
            scheduler.set_timesteps(args.steps, device=device)
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
                        noise = uncond + args.guidance * (conditioned - uncond)
                    latents = scheduler.step(noise, t, latents, **extra_step_kwargs).prev_sample
                if n > 1:
                    latents = torch.cat([latents[k] for k in range(n)], dim=1).unsqueeze(
                        0
                    )  # back to (1, C, n*window, h, w)
                decoded = pipeline.decode_latents(latents)
                decoded = pipeline.paste_surrounding_pixels_back(
                    decoded, pixel_values, 1 - masks, device, dtype
                )
            for k, real in enumerate(reals):
                outputs.append(decoded[k * window : k * window + real])
        return torch.cat(outputs)

    capture = cv2.VideoCapture(video)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    if args.spans_only:
        # the finisher (backend/rendering/finish.py) pastes the cached spans into the picture and encodes once, so nothing
        # is written here but the spans and a manifest; cached spans are not even decoded
        if not cache_dir:
            raise SystemExit("--spans-only needs --cache")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        manifest, started, done, reused = [], time.time(), 0, 0
        todo = []
        for first, end in spans:
            span = (first, end)
            path = os.path.join(cache_dir, f"span_{keys[span]}.mp4")
            manifest.append({"first": first, "end": end, "key": keys[span], "file": path})
            if os.path.isfile(path) and cached_frames(span) is not None:
                reused += end - first
            else:
                todo.append((span, path))

        def write_manifest(complete):
            """spans.json, rewritten as spans land, so a preview can be built from what exists so far."""
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

        write_manifest(False)

        def prepare(span):
            """CPU, ahead of the GPU: decode the span's frames and align the faces."""
            first, end = span
            reader = cv2.VideoCapture(video)
            reader.set(cv2.CAP_PROP_POS_FRAMES, first)
            frames = []
            for _ in range(end - first):
                ok, frame = reader.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            reader.release()
            if len(frames) != end - first:
                raise RuntimeError(f"could not read frames {first}-{end} of {video}")
            return frames, align_faces(frames, first)

        def finish(span, path, frames, generated, aligned):
            """CPU, behind the GPU: paste the faces back and store the span."""
            _, matrices, sizes, restorer = aligned
            rendered = paste_back(frames, generated, matrices, sizes, restorer)
            store_span(path, [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rendered], fps)
            return span[1] - span[0]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            ahead = pool.submit(prepare, todo[0][0]) if todo else None
            pending = []
            for index, (span, path) in enumerate(todo):
                frames, aligned = ahead.result()
                ahead = pool.submit(prepare, todo[index + 1][0]) if index + 1 < len(todo) else None
                generated = diffuse(
                    aligned[0], span[0]
                )  # the GPU part; the next span aligns meanwhile
                pending.append(pool.submit(finish, span, path, frames, generated, aligned))
                for future in [f for f in pending if f.done()]:
                    done += future.result()
                    pending.remove(future)
                write_manifest(False)
                print(
                    f"Rendered {done} of {total_target} frames ({time.time() - started:.0f}s)",
                    flush=True,
                )
            for future in pending:
                done += future.result()
        write_manifest(True)
        print(
            f"DONE frames={total_frames} synced={done + reused} rendered={done} reused={reused} seconds={time.time() - started:.1f}",
            flush=True,
        )
        return 0
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
            str(min(args.crf, 8)),
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        stdin=subprocess.PIPE,
    )
    position, done, reused, started = 0, 0, 0, time.time()

    def copy_until(stop):
        nonlocal position
        while position < stop:
            ok, frame = capture.read()
            if not ok:
                return False
            writer.stdin.write(frame.tobytes())
            position += 1
        return True

    for index, (first, end) in enumerate(spans):
        if not copy_until(first):
            break
        frames = []
        for _ in range(end - first):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            break
        span = (first, end)
        output_frames = cached_frames(span) if len(frames) == end - first else None
        if output_frames is not None:
            reused += len(frames)
        else:
            rendered = render([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], first)
            output_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rendered]
            if cache_dir and len(frames) == end - first:
                store_span(os.path.join(cache_dir, f"span_{keys[span]}.mp4"), output_frames, fps)
        for frame in output_frames:
            writer.stdin.write(frame.tobytes())
        position += len(frames)
        done += len(frames)
        print(
            f"Rendered {done} of {total_target} frames ({time.time() - started:.0f}s)", flush=True
        )
    copy_until(10**9)
    capture.release()
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg could not write the lip-synced video")
    print(
        f"DONE frames={position} synced={done} reused={reused} seconds={time.time() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
