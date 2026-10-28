"""Bounded-memory Wav2Lip rendering with exact frame timing and SFD detection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import cv2
import numpy as np
import torch
import audio
from face_tracking import BoxSmoother, FaceTracker, blend_mouth, mel_window
from models import Wav2Lip


def render(args):
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    device = 'cuda' if args.runtime == 'torch' and torch.cuda.is_available() else 'cpu'
    batch_size = args.wav2lip_batch_size or (64 if device == 'cuda' else 8)
    targets = np.load(args.targets)['bbox'] if args.targets else None
    detector = None if targets is not None else FaceTracker(device, interval=args.face_det_interval)
    smoother = BoxSmoother(window=args.smooth_window)
    enhancer = None
    if args.enhance_weights:
        from enhance import MouthEnhancer
        enhancer = MouthEnhancer(Path(args.enhance_weights), args.runtime, args.openvino_device)
    model = Wav2Lip().to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict({key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()})
    del checkpoint
    model.eval()
    if args.runtime == 'openvino':
        from inference_runtime import MouthInference, FaceInference
        model = MouthInference(model, args.checkpoint_path, args.openvino_device)
        if detector is not None:
            face_model = detector.detector.face_detector
            face_checkpoint = Path(__file__).parent / 'face_detection/detection/sfd/s3fd.pth'
            face_model.face_detector = FaceInference(face_model.face_detector, face_checkpoint, args.openvino_device)
        report_device = f'openvino:{model.device}'
    else:
        report_device = device
    wav = audio.load_wav(args.audio, 16000)
    if not len(wav):
        raise ValueError('The audio input is empty')
    mel = audio.melspectrogram(wav)
    if not np.isfinite(mel).all():
        raise ValueError('Invalid audio features')
    capture = cv2.VideoCapture(args.face)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError('Cannot decode source video')
    started = time.perf_counter()
    report = {'frames': 0, 'synced_frames': 0, 'no_face': 0, 'not_speaking': 0,
              'multiple_faces': 0, 'small_face': 0, 'fps': fps,
              'detector': 'studio targets' if args.targets else 'SFD'}
    report['face_detection_interval'] = args.face_det_interval
    report['mouth_enhanced'] = bool(args.enhance_weights)
    report['runtime'] = args.runtime
    report['device'] = report_device
    output = Path(args.outfile).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='render-') as temporary:
        intermediate = Path(temporary) / 'frames.mkv'
        try:
            while True:
                frames = []
                for _ in range(batch_size):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if args.resize_factor > 1:
                        frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
                    frames.append(frame)
                if not frames:
                    break
                if writer is None:
                    height, width = frames[0].shape[:2]
                    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'FFV1'), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create lossless video intermediate')
                detections = []
                if targets is not None:
                    # The studio decided which face speaks in each frame (active-speaker detection).
                    for offset in range(len(frames)):
                        index = args.target_offset + report['frames'] + offset
                        box = targets[index] if index < len(targets) else None
                        if box is None or not np.isfinite(box).all():
                            detections.append((None, 'not_speaking'))
                            continue
                        h, w = frames[offset].shape[:2]
                        x1, y1, x2, y2 = (int(round(v / args.resize_factor)) for v in box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2 + 10)
                        detections.append(((x1, y1, x2, y2), None) if x2 - x1 >= 24 and y2 - y1 >= 24 else (None, 'small_face'))
                else:
                    for offset in range(0, len(frames), args.face_det_batch_size):
                        detections.extend(detector.detect(frames[offset:offset + args.face_det_batch_size]))
                detections = [(smoother.apply(box), reason) for box, reason in detections]
                images, windows, indices = [], [], []
                for index, (box, reason) in enumerate(detections):
                    if box is None:
                        report[reason] += 1
                        continue
                    x1, y1, x2, y2 = box
                    crop = cv2.resize(frames[index][y1:y2, x1:x2], (96, 96))
                    masked = crop.copy()
                    masked[48:] = 0
                    images.append(np.concatenate((masked, crop), axis=2) / 255.0)
                    windows.append(mel_window(mel, report['frames'] + index, fps))
                    indices.append(index)
                if indices:
                    image_tensor = torch.from_numpy(np.asarray(images, dtype=np.float32).transpose(0, 3, 1, 2)).to(device)
                    mel_tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32)[:, None]).to(device)
                    with torch.inference_mode():
                        predictions = model(mel_tensor, image_tensor).cpu().numpy().transpose(0, 2, 3, 1) * 255
                    if enhancer is not None:
                        predictions = enhancer(np.clip(predictions, 0, 255).astype(np.uint8))
                    for index, predicted in zip(indices, predictions):
                        x1, y1, x2, y2 = detections[index][0]
                        region = frames[index][y1:y2, x1:x2]
                        frames[index][y1:y2, x1:x2] = blend_mouth(region, predicted)
                    report['synced_frames'] += len(indices)
                for frame in frames:
                    writer.write(frame)
                report['frames'] += len(frames)
                print(f"Rendered {report['frames']} frames", flush=True)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        if not report['frames']:
            raise RuntimeError('Source video contains no decoded frames')
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(intermediate), '-i', args.audio,
                        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-af', 'apad',
                        '-t', str(report['frames'] / fps), '-movflags', '+faststart', str(output)], check=True)
    report['render_seconds'] = round(time.perf_counter() - started, 2)
    Path(args.report or str(output) + '.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--face', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--outfile', required=True)
    parser.add_argument('--report')
    parser.add_argument('--resize_factor', type=int, default=1)
    parser.add_argument('--face_det_batch_size', type=int, default=4)
    parser.add_argument('--face_det_interval', type=int, default=1)
    parser.add_argument('--runtime', choices=['torch', 'openvino'], default='torch')
    parser.add_argument('--openvino_device', default='GPU')
    parser.add_argument('--wav2lip_batch_size', type=int, default=0)
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--smooth_window', type=int, default=5, help='frames averaged for the face box (1 disables)')
    parser.add_argument('--enhance_weights', default='', help='realesr-general-x4v3.pth path; enables mouth super-resolution')
    parser.add_argument('--targets', default='', help='per-frame target boxes (npz from the studio); replaces face detection')
    parser.add_argument('--target_offset', type=int, default=0, help='index of this clip\'s first frame in the targets')
    args = parser.parse_args()
    if args.resize_factor < 1 or args.face_det_batch_size < 1:
        parser.error('Resize factor and detection batch size must be positive')
    render(args)
