# Copied verbatim from /projects/data/ttsteam/avnish/tts_audio/normalize_volume.py (the normaliser behind audio_generations_peak_normalized, 2026-10-20).
# Do not edit here; update from the source.
#!/usr/bin/env python3
"""
Volume normalization for TTS audio.

Fixes two problems:
  1. Within-file loudness DRIFT  (e.g. speech that "falls flat" towards the end)
  2. Between-file loudness inconsistency

Main mode is `agc` (default): a slow, speech-gated automatic-gain-control that
tracks the long-term speech envelope and flattens it to a constant target level
without squashing natural syllable-level dynamics and without boosting silence /
noise floor. A final look-ahead peak limiter prevents clipping.

`global` mode just applies a single gain so the whole file hits the target
loudness (useful when a file is uniform but too quiet/loud).

Only needs: numpy, scipy, soundfile.

Examples
--------
  # Normalize one file -> writes <name>_norm.wav next to it
  python normalize_volume.py ta/...01511.wav

  # Normalize a whole language dir recursively, mirror into ta_norm/
  python normalize_volume.py ta -o ta_norm

  # Overwrite in place (careful)
  python normalize_volume.py ta --inplace

  # Just report drift, write nothing
  python normalize_volume.py ta --dry-run

  # Tune target / max boost
  python normalize_volume.py ta --target-rms -20 --max-gain 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d, maximum_filter1d

EPS = 1e-12


def db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


def undb(d):
    return 10.0 ** (d / 20.0)


def to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def speech_gated_rms_db(mono, sr, frame=0.05):
    """Overall speech-only RMS in dBFS (ignores silence / gaps)."""
    fs = max(1, int(frame * sr))
    p = uniform_filter1d(mono.astype(np.float64) ** 2, fs)[::fs]
    fdb = 10.0 * np.log10(p + EPS)
    ref = np.percentile(fdb, 95)
    floor = max(ref - 35.0, -55.0)
    speech = p[fdb > floor]
    if speech.size == 0:
        speech = p
    return 10.0 * np.log10(speech.mean() + EPS)


def drift_db(mono, sr):
    """Loudness of last 10% minus first 10% of speech (negative => fades out)."""
    n = len(mono)
    head = mono[: max(1, n // 10)]
    tail = mono[-max(1, n // 10):]
    return speech_gated_rms_db(tail, sr) - speech_gated_rms_db(head, sr)


def compute_agc_gain_db(mono, sr, target_db, window, max_gain, min_gain,
                        smooth, frame=0.02, speech_range=35.0, abs_floor=-55.0,
                        boost_only=False, target_pct=90.0, noise_ceiling=-50.0):
    """
    Build a per-sample gain curve (in dB) that flattens the long-term speech
    envelope to `target_db`.

    window      : seconds over which the long-term speech level is measured
    smooth      : seconds of additional smoothing applied to the gain curve
    boost_only  : if True, gain is clamped to >= 0 dB (good parts never lowered)

    target_db / max_gain / min_gain: pass None to derive them from the audio.
        target_db -> `target_pct` percentile of speech (the file's good level)
        max_gain  -> what's needed to lift the quietest speech to target,
                     capped so boosted background noise stays below noise_ceiling
        min_gain  -> 0 for boost; else enough to pull loud peaks down to target
    A non-None value overrides the auto computation.

    Returns (per_sample_gain_db, target_db, max_gain, min_gain).
    """
    n = len(mono)
    fs = max(1, int(frame * sr))                 # samples per analysis frame
    sq = mono.astype(np.float64) ** 2

    # frame power
    p = uniform_filter1d(sq, fs)[::fs]
    nf = len(p)
    fdb = 10.0 * np.log10(p + EPS)

    # speech mask (relative to this file's loud reference + absolute floor)
    ref = np.percentile(fdb, 95)
    floor = max(ref - speech_range, abs_floor)
    mask = (fdb > floor).astype(np.float64)
    if mask.sum() == 0:
        return None  # nothing to normalize

    # speech-gated local power: average power over speech frames within `window`
    W = max(1, int(round(window / frame)))
    num = uniform_filter1d(p * mask, W)
    den = uniform_filter1d(mask, W)
    valid = den > 1e-3
    local_db = np.full(nf, np.nan)
    local_db[valid] = 10.0 * np.log10(num[valid] / den[valid] + EPS)

    # ---- per-file dynamic parameters (any arg left None is computed here) ----
    spk = local_db[valid]                       # long-term speech levels (dBFS)

    # target = the file's own "good" loudness (high percentile of speech)
    if target_db is None:
        target_db = float(np.percentile(spk, target_pct))

    # noise floor from non-speech frames -> how much we can boost safely.
    # use a low percentile (the true silence floor), not the near-threshold dips.
    noise = fdb[mask < 0.5]
    noise_db = float(np.percentile(noise, 25)) if noise.size > 0.02 * nf else -90.0
    headroom = max(0.0, noise_ceiling - noise_db)   # boost we can add before
                                                    # background reaches the ceiling
    # max boost = enough to lift the quietest speech to target, capped by headroom
    if max_gain is None:
        deficit = target_db - float(np.percentile(spk, 2))
        max_gain = float(min(max(deficit, 0.0), headroom))

    # max attenuation = enough to pull loud peaks down to target (agc only)
    if min_gain is None:
        min_gain = 0.0 if boost_only else \
            float(np.clip(target_db - np.percentile(spk, 98), -15.0, 0.0))
    if boost_only:
        min_gain = max(min_gain, 0.0)               # never lower the good parts

    # desired gain on speech windows
    gain_db = target_db - local_db
    gain_db = np.clip(gain_db, min_gain, max_gain)

    # silence windows (no speech nearby): interpolate gain from neighbours so
    # gaps ride along with surrounding speech instead of being boosted.
    idx = np.arange(nf)
    good = ~np.isnan(gain_db)
    gain_db = np.interp(idx, idx[good], gain_db[good])

    # smooth the gain curve (avoid pumping)
    Ws = max(1, int(round(smooth / frame)))
    gain_db = uniform_filter1d(gain_db, Ws)

    # upsample frame gain -> per sample
    centers = idx * fs + fs // 2
    return np.interp(np.arange(n), centers, gain_db), target_db, max_gain, min_gain


def peak_limit(y, peak_db, sr, attack=0.005, release=0.05):
    """Look-ahead peak limiter: keep |y| <= peak, smoothly, then hard-clip."""
    peak_lin = undb(peak_db)
    mono = to_mono(np.abs(y))
    env = maximum_filter1d(mono, max(1, int(attack * sr)))      # look-ahead peak
    gr = np.minimum(1.0, peak_lin / (env + EPS))                # gain reduction
    gr = uniform_filter1d(gr, max(1, int(release * sr)))        # smooth release
    gr = gr[:, None] if y.ndim > 1 else gr
    y = y * gr
    return np.clip(y, -peak_lin, peak_lin)


def normalize_file(path, args):
    x, sr = sf.read(str(path), always_2d=False)
    x = x.astype(np.float64)
    mono = to_mono(x)

    in_rms = speech_gated_rms_db(mono, sr)
    in_drift = drift_db(mono, sr)
    in_peak = db(np.max(np.abs(mono)) + EPS)

    if args.mode == "global":
        eff_target = args.target_rms if args.target_rms is not None else -20.0
        eff_max = args.max_gain if args.max_gain is not None else 20.0
        eff_min = args.min_gain if args.min_gain is not None else -15.0
        g = np.clip(eff_target - in_rms, eff_min, eff_max)
        y = x * undb(g)
        gain_lo = gain_hi = g
    else:  # agc / boost  -- target/max/min derived per file unless overridden
        out = compute_agc_gain_db(
            mono, sr, args.target_rms, args.window, args.max_gain, args.min_gain,
            args.smooth, boost_only=(args.mode == "boost"),
            target_pct=args.target_pct, noise_ceiling=args.noise_ceiling)
        if out is None:
            print(f"  SKIP (no speech detected): {path}")
            return None
        gdb, eff_target, eff_max, eff_min = out
        gain_lo, gain_hi = float(gdb.min()), float(gdb.max())
        glin = undb(gdb)
        y = x * (glin[:, None] if x.ndim > 1 else glin)

    y = peak_limit(y, args.peak, sr)

    out_mono = to_mono(y)
    out_rms = speech_gated_rms_db(out_mono, sr)
    out_drift = drift_db(out_mono, sr)
    out_peak = db(np.max(np.abs(out_mono)) + EPS)

    print(f"  {path.name}")
    print(f"    rms  {in_rms:6.1f} -> {out_rms:6.1f} dBFS   "
          f"drift {in_drift:+5.1f} -> {out_drift:+5.1f} dB   "
          f"peak {in_peak:5.1f} -> {out_peak:5.1f} dBFS   "
          f"gain [{gain_lo:+.1f},{gain_hi:+.1f}]   "
          f"auto: tgt {eff_target:.1f} cap [{eff_min:+.1f},{eff_max:+.1f}]")

    if args.dry_run:
        return out_rms

    out_path = resolve_output(path, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subtype = sf.info(str(path)).subtype
    sf.write(str(out_path), y, sr, subtype=subtype)
    return out_rms


def resolve_output(path, args):
    if args.inplace:
        return path
    if args.output_dir:
        out_root = Path(args.output_dir)
        rel = None
        if args.in_root is not None:
            try:
                rel = path.relative_to(args.in_root)
            except ValueError:
                rel = None
        return out_root / (rel if rel is not None else path.name)
    return path.with_name(path.stem + args.suffix + path.suffix)


def gather(paths, ext):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.rglob(f"*{ext}")))
        elif p.is_file():
            out.append(p)
        else:
            print(f"  WARN: not found: {p}", file=sys.stderr)
    # don't re-process our own outputs
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="wav file(s) or directory(ies)")
    ap.add_argument("--mode", choices=["agc", "boost", "global"], default="agc",
                    help="agc = flatten drift to a fixed target (default); "
                         "boost = lift quiet/drifting parts up to the file's good "
                         "level, never lowering loud parts; global = single gain")
    ap.add_argument("--target-rms", type=float, default=None,
                    help="OVERRIDE target speech RMS in dBFS. Default: auto per "
                         "file (the file's own 'good' level); global uses -20")
    ap.add_argument("--target-pct", type=float, default=90.0,
                    help="percentile of speech level used as the auto target "
                         "(default 90)")
    ap.add_argument("--window", type=float, default=4.0,
                    help="AGC tracking window in seconds (default 4)")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="AGC gain smoothing in seconds (default 1)")
    ap.add_argument("--max-gain", type=float, default=None,
                    help="OVERRIDE max boost in dB. Default: auto per file "
                         "(enough to reach target, capped by noise headroom)")
    ap.add_argument("--min-gain", type=float, default=None,
                    help="OVERRIDE max attenuation in dB. Default: auto "
                         "(0 for boost; pull peaks to target for agc)")
    ap.add_argument("--noise-ceiling", type=float, default=-50.0,
                    help="don't boost background noise above this dBFS "
                         "(governs the auto max-gain cap, default -50)")
    ap.add_argument("--peak", type=float, default=-1.0,
                    help="output peak ceiling in dBFS (default -1)")
    ap.add_argument("--ext", default=".wav", help="extension to match in dirs")
    ap.add_argument("-o", "--output-dir", help="mirror outputs into this dir")
    ap.add_argument("--suffix", default="_norm",
                    help="suffix when writing next to original (default _norm)")
    ap.add_argument("--inplace", action="store_true", help="overwrite originals")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyze and report only, write nothing")
    args = ap.parse_args()

    args.in_root = Path(args.paths[0]) if len(args.paths) == 1 and Path(args.paths[0]).is_dir() else None

    files = gather(args.paths, args.ext)
    # skip already-normalized siblings unless explicitly given
    if not args.inplace and not args.output_dir:
        files = [f for f in files if not f.stem.endswith(args.suffix)]
    if not files:
        print("No audio files found.")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(files)} file(s), "
          f"mode={args.mode}, target={args.target_rms} dBFS\n")
    for f in files:
        try:
            normalize_file(f, args)
        except Exception as e:
            print(f"  ERROR {f}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
