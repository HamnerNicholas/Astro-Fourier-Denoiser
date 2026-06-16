#!/usr/bin/env python3
"""
Fourier coherence denoiser for aligned astrophotography subframes.

Usage:
    python fourier_denoise_engine_fits.py ./input_frames ./denoised_frames

Install dependencies:
    pip install numpy imageio astropy

Supported input:
    - PNG, JPG, JPEG, TIF, TIFF
    - FIT, FITS, FTS
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import imageio.v3 as iio

import cv2

from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter

try:
    from astropy.io import fits
except ImportError:
    fits = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FITS_EXTS = {".fit", ".fits", ".fts"}
SUPPORTED_EXTS = IMAGE_EXTS | FITS_EXTS

def debayer_frame(frame: np.ndarray, pattern="RGGB") -> np.ndarray:
    """
    Convert mono Bayer frame into RGB.
    """

    frame = frame.astype(np.uint16)

    patterns = {
        "RGGB": cv2.COLOR_BAYER_RG2RGB,
        "BGGR": cv2.COLOR_BAYER_BG2RGB,
        "GRBG": cv2.COLOR_BAYER_GR2RGB,
        "GBRG": cv2.COLOR_BAYER_GB2RGB,
    }

    return cv2.cvtColor(frame, patterns[pattern]).astype(np.float32)

def frame_to_luminance(frame):
    if frame.ndim == 2:
        return frame

    return (
        0.2126 * frame[..., 0]
        + 0.7152 * frame[..., 1]
        + 0.0722 * frame[..., 2]
    )


def align_frames_translation(frames):
    """
    Align all frames to frame 0 using phase correlation.
    Works for mono FITS and RGB data.
    """

    reference = normalize_for_alignment(frame_to_luminance(frames[0]))

    aligned = [frames[0]]

    print("Aligning frames...")

    for i in range(1, len(frames)):
        moving = normalize_for_alignment(frame_to_luminance(frames[i]))

        shift_yx, error, phase = phase_cross_correlation(
            reference,
            moving,
            upsample_factor=20,
            normalization=None
        )

        print(
            f"Frame {i+1}: "
            f"dy={shift_yx[0]:.2f}, "
            f"dx={shift_yx[1]:.2f}"
        )

        frame = frames[i]

        if frame.ndim == 2:
            aligned_frame = ndi_shift(
                frame,
                shift=shift_yx,
                order=3,
                mode="constant",
                cval=0
            )

        else:
            aligned_frame = np.empty_like(frame)

            for ch in range(frame.shape[-1]):
                aligned_frame[..., ch] = ndi_shift(
                    frame[..., ch],
                    shift=shift_yx,
                    order=3,
                    mode="constant",
                    cval=0
                )

        aligned.append(aligned_frame)

    return aligned

def normalize_for_alignment(img):
    img = img.astype(np.float32)

    low, high = np.percentile(img, [1, 99.8])
    img = np.clip(img, low, high)

    img = img - np.median(img)

    scale = np.std(img)
    if scale > 0:
        img = img / scale

    return img

def find_frame_files(input_dir: Path) -> list[Path]:
    files = [
    p for p in sorted(input_dir.iterdir())
    if (
        p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and "autosave" not in p.name.lower()
        and "denoised" not in p.name.lower()
        and "difference" not in p.name.lower()
    )
]
    if not files:
        raise FileNotFoundError(f"No supported frames found in: {input_dir}")
    return files


def is_fits_file(path: Path) -> bool:
    return path.suffix.lower() in FITS_EXTS


def normalize_fits_shape(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    data = np.squeeze(data)

    if data.ndim == 2:
        return data

    if data.ndim == 3:
        if data.shape[0] == 3:
            return np.moveaxis(data, 0, -1)
        if data.shape[-1] == 3:
            return data

    raise ValueError(f"Unsupported FITS data shape: {data.shape}")


def load_fits(path: Path) -> tuple[np.ndarray, np.dtype, dict]:
    if fits is None:
        raise ImportError("FITS support requires astropy. Install it with: pip install astropy")

    with fits.open(path) as hdul:
        hdu_index = None

        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                hdu_index = i
                break

        if hdu_index is None:
            raise ValueError(f"No image data found in FITS file: {path.name}")

        hdu = hdul[hdu_index]
        data = normalize_fits_shape(hdu.data)
        header = hdu.header.copy()
        bayer_pattern = header.get("BAYERPAT", None)

    original_dtype = data.dtype
    return data.astype(np.float32), original_dtype, {
        "header": header,
        "bayer_pattern": bayer_pattern
    }

def load_standard_image(path: Path) -> tuple[np.ndarray, np.dtype, dict]:
    img = iio.imread(path)
    original_dtype = img.dtype

    if img.ndim == 2:
        return img.astype(np.float32), original_dtype, {}

    if img.ndim == 3 and img.shape[-1] > 3:
        img = img[..., :3]

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape for {path.name}: {img.shape}")

    return img.astype(np.float32), original_dtype, {}


def load_frame(path: Path) -> tuple[np.ndarray, np.dtype, dict]:
    if is_fits_file(path):
        return load_fits(path)
    return load_standard_image(path)


def dtype_range(dtype: np.dtype) -> tuple[float, float]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return -np.inf, np.inf


def save_standard_image(path: Path, img: np.ndarray, dtype: np.dtype) -> None:
    lo, hi = dtype_range(dtype)
    img = np.clip(img, lo, hi)

    if np.issubdtype(dtype, np.integer):
        img = np.rint(img).astype(dtype)
    else:
        img = img.astype(dtype)

    iio.imwrite(path, img)


def save_fits(path: Path, img: np.ndarray, dtype: np.dtype, metadata: dict) -> None:
    if fits is None:
        raise ImportError("FITS support requires astropy. Install it with: pip install astropy")

    header = metadata.get("header")

    if img.ndim == 3 and img.shape[-1] == 3:
        data = np.moveaxis(img, -1, 0)
    else:
        data = img

    if np.issubdtype(dtype, np.integer):
        lo, hi = dtype_range(dtype)
        data = np.clip(data, lo, hi)
        data = np.rint(data).astype(dtype)
    else:
        data = data.astype(np.float32)

    fits.writeto(path, data, header=header, overwrite=True)


def save_frame(path: Path, img: np.ndarray, dtype: np.dtype, metadata: dict) -> None:
    if is_fits_file(path):
        save_fits(path, img, dtype, metadata)
    else:
        save_standard_image(path, img, dtype)


def frame_to_luminance(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame

    return (
        0.2126 * frame[..., 0]
        + 0.7152 * frame[..., 1]
        + 0.0722 * frame[..., 2]
    )


def build_luminance_stack(frames: list[np.ndarray]) -> np.ndarray:
    return np.stack([frame_to_luminance(frame) for frame in frames], axis=0)


def median_absolute_deviation(image: np.ndarray) -> float:
    values = image.astype(np.float32).ravel()
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(mad)


def frame_noise_metrics(frame: np.ndarray) -> dict[str, float]:
    """
    Calculate full-frame MAD and background-only MAD.
    Background is estimated as the dimmest 80% of luminance pixels.
    """

    lum = frame_to_luminance(frame)

    full_mad = median_absolute_deviation(lum)
    full_sigma = 1.4826 * full_mad

    cutoff = np.percentile(lum, 80)
    background = lum[lum <= cutoff]

    background_mad = median_absolute_deviation(background)
    background_sigma = 1.4826 * background_mad

    return {
        "mad_luminance": full_mad,
        "sigma_est_luminance": full_sigma,
        "background_mad": background_mad,
        "background_sigma_est": background_sigma,
    }

def ensure_rgb_frame(frame: np.ndarray) -> np.ndarray:
    """
    Convert mono frames to RGB-like 3-channel frames for per-channel processing.
    """

    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)

    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame

    raise ValueError(f"Unsupported frame shape for RGB processing: {frame.shape}")

def build_coherence_mask(
    frames: list[np.ndarray],
    gamma: float = 1.5,
    floor: float = 0.25,
    eps: float = 1e-8,
    per_channel: bool = True,
) -> np.ndarray:
    """
    If per_channel=True:
        returns mask with shape (H, W, 3)

    If per_channel=False:
        returns luminance mask with shape (H, W)
    """
    
    if not per_channel:
        lum = build_luminance_stack(frames)
        F = np.fft.fft2(lum, axes=(1, 2))

        coherence = np.abs(np.mean(F, axis=0)) / (np.mean(np.abs(F), axis=0) + eps)
        coherence = np.clip(coherence, 0.0, 1.0)

        mask = floor + (1.0 - floor) * (coherence ** gamma)
        return np.clip(mask, floor, 1.0).astype(np.float32)

    rgb_frames = [ensure_rgb_frame(frame) for frame in frames]

    masks = []

    channel_gammas = [gamma, gamma, gamma]
    channel_floors = [floor, floor, floor]

    for ch in range(3):
        channel_stack = np.stack([frame[..., ch] for frame in rgb_frames], axis=0)
        F = np.fft.fft2(channel_stack, axes=(1, 2))

        coherence = np.abs(np.mean(F, axis=0)) / (np.mean(np.abs(F), axis=0) + eps)
        coherence = np.clip(coherence, 0.0, 1.0)

        gamma_ch = channel_gammas[ch]
        floor_ch = channel_floors[ch]

        gamma_ch = channel_gammas[ch]
        floor_ch = channel_floors[ch]

        mask = floor_ch + (
            1.0 - floor_ch
        ) * (coherence ** gamma_ch)

        masks.append(
            np.clip(mask, floor_ch, 1.0).astype(np.float32)
        )

        print(f"Channel {ch}: gamma={gamma_ch}, floor={floor_ch}")


    return np.stack(masks, axis=-1)



def apply_mask_to_frame(frame: np.ndarray, rgb_mask: np.ndarray, lum_mask: np.ndarray | None = None) -> np.ndarray:
    if frame.ndim == 2:
        F = np.fft.fft2(frame)

        if lum_mask is not None:
            mono_mask = lum_mask
        elif rgb_mask.ndim == 3:
            mono_mask = np.mean(rgb_mask, axis=-1)
        else:
            mono_mask = rgb_mask

        return np.fft.ifft2(F * mono_mask).real.astype(np.float32)

    output = np.empty_like(frame, dtype=np.float32)

    for ch in range(3):
        F = np.fft.fft2(frame[..., ch])

        if rgb_mask.ndim == 3:
            channel_mask = rgb_mask[..., ch]
        else:
            channel_mask = rgb_mask

        output[..., ch] = np.fft.ifft2(F * channel_mask).real

    return output

def apply_radial_gate(mask: np.ndarray, cutoff: float = 0.12, softness: float = 0.15) -> np.ndarray:
    if mask.ndim == 3:
        gated = np.empty_like(mask)
        for ch in range(3):
            gated[..., ch] = apply_radial_gate(mask[..., ch], cutoff, softness)
        return gated

    h, w = mask.shape

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)

    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)

    gate = np.clip((radius - cutoff) / softness, 0.0, 1.0)

    gated_mask = 1.0 - gate * (1.0 - mask)

    return np.clip(gated_mask, 0.0, 1.0).astype(np.float32)

def denoise_folder(
    input_dir: Path,
    output_dir: Path,
    gamma: float,
    floor: float,
    batch_limit: int | None = None,
    mask_sigma: float = 5.0,
    radial_cutoff: float = 0.12,
    radial_softness: float = 0.15,
    save_difference: bool = True,
    save_masks: bool = True,
) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_frame_files(input_dir)
    if batch_limit is not None:
        files = files[:batch_limit]

    print(f"Found {len(files)} frame(s).")
    print("Loading frames...")

    frames: list[np.ndarray] = []
    dtypes: list[np.dtype] = []
    metadata_list: list[dict] = []
    shape = None

    use_per_channel = True

    for path in files:
        frame, dtype, metadata = load_frame(path)

        if frame.ndim == 2:
            pattern = metadata.get("bayer_pattern")

            if pattern:
                print(f"Debayering {path.name} ({pattern})")
                frame = debayer_frame(frame, pattern)
            else:
                print(
                    f"WARNING: {path.name} is mono but has no BAYERPAT. "
                    "Using luminance mask mode."
                )
                use_per_channel = False

            lum_shape = frame_to_luminance(frame).shape

            if shape is None:
                shape = lum_shape
            elif lum_shape != shape:
                raise ValueError(
                    f"All frames must have the same luminance shape. "
                    f"{path.name} has {lum_shape}, expected {shape}."
                    )

        frames.append(frame)
        dtypes.append(dtype)
        metadata_list.append(metadata)

    frames = align_frames_translation(frames)

    print("Building RGB per-channel mask...")
    rgb_mask = build_coherence_mask(
        frames,
        gamma=gamma,
        floor=floor,
        per_channel=True
    )

    print("Building shared luminance mask...")
    lum_mask = build_coherence_mask(
        frames,
        gamma=gamma,
        floor=floor,
        per_channel=False
    )

    print("Smoothing frequency masks...")
    channel_sigmas = [
        mask_sigma,        # Red
        mask_sigma,        # Green
        mask_sigma * 2.0,  # Blue
    ]

    for ch in range(3):
        rgb_mask[..., ch] = gaussian_filter(
            rgb_mask[..., ch],
            sigma=channel_sigmas[ch]
        )

    lum_mask = gaussian_filter(lum_mask, sigma=mask_sigma)

    print("Applying radial frequency gate...")
    rgb_mask = apply_radial_gate(
        rgb_mask,
        cutoff=radial_cutoff,
        softness=radial_softness
    )

    lum_mask = apply_radial_gate(
        lum_mask,
        cutoff=radial_cutoff,
        softness=radial_softness
    )

    # Save masks for inspection in Siril
    print(
        "R:",
        rgb_mask[...,0].min(),
        rgb_mask[...,0].max(),
        rgb_mask[...,0].mean(),
        rgb_mask[...,0].std()
    )

    print(
        "G:",
        rgb_mask[...,1].min(),
        rgb_mask[...,1].max(),
        rgb_mask[...,1].mean(),
        rgb_mask[...,1].std()
    )

    print(
        "B:",
        rgb_mask[...,2].min(),
        rgb_mask[...,2].max(),
        rgb_mask[...,2].mean(),
        rgb_mask[...,2].std()
    )

    if save_masks:
        fits.writeto(
            output_dir / "mask_R.fit",
            rgb_mask[..., 0].astype(np.float32),
            overwrite=True
        )

        fits.writeto(
            output_dir / "mask_G.fit",
            rgb_mask[..., 1].astype(np.float32),
            overwrite=True
        )

        fits.writeto(
            output_dir / "mask_B.fit",
            rgb_mask[..., 2].astype(np.float32),
            overwrite=True
        )

        fits.writeto(
            output_dir / "mask_L.fit",
            lum_mask.astype(np.float32),
            overwrite=True
        )

    print("Denoising frames...")
    metrics_rows = []

    for idx, (src_path, dtype, metadata) in enumerate(
        zip(files, dtypes, metadata_list),
        start=1,
    ):
        original = frames[idx - 1]
        denoised = apply_mask_to_frame(original, rgb_mask, lum_mask)

        difference = original - denoised

        before = frame_noise_metrics(original)
        after = frame_noise_metrics(denoised)
        
        background_mad_before = before["background_mad"]
        background_mad_after = after["background_mad"]

        background_mad_reduction_pct = (
            100.0 * (background_mad_before - background_mad_after) / background_mad_before
            if background_mad_before > 0 else 0.0)

        mad_before = before["mad_luminance"]
        mad_after = after["mad_luminance"]
        sigma_before = before["sigma_est_luminance"]
        sigma_after = after["sigma_est_luminance"]

        mad_reduction_pct = (
            100.0 * (mad_before - mad_after) / mad_before
            if mad_before > 0 else 0.0
        )

        sigma_reduction_pct = (
            100.0 * (sigma_before - sigma_after) / sigma_before
            if sigma_before > 0 else 0.0
        )

        out_name = f"{src_path.stem}_denoised{src_path.suffix}"
        out_path = output_dir / out_name

        save_frame(out_path, denoised, dtype, metadata)

        if save_difference:
            diff_name = f"{src_path.stem}_difference{src_path.suffix}"
            diff_path = output_dir / diff_name
            save_frame(diff_path, difference, np.float32, metadata)

        metrics_rows.append({
            "frame": src_path.name,
            "output": out_name,
            "mad_before": mad_before,
            "mad_after": mad_after,
            "mad_reduction_pct": mad_reduction_pct,
            "sigma_est_before": sigma_before,
            "sigma_est_after": sigma_after,
            "sigma_est_reduction_pct": sigma_reduction_pct,
            "background_mad_before": background_mad_before,
            "background_mad_after": background_mad_after,
            "background_mad_reduction_pct": background_mad_reduction_pct,
        })

        print(
            f"[{idx}/{len(files)}] saved {out_path.name} | "
            f"MAD {mad_before:.4f} -> {mad_after:.4f} "
            f"({mad_reduction_pct:.2f}% reduction)"
        )

    csv_path = output_dir / "denoise_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "output",
                "mad_before",
                "mad_after",
                "mad_reduction_pct",
                "sigma_est_before",
                "sigma_est_after",
                "sigma_est_reduction_pct",
                "background_mad_before",
                "background_mad_after",
                "background_mad_reduction_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"Metrics written to: {csv_path}")
    print(f"Done. Denoised frames written to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise aligned astrophotography subframes using Fourier coherence."
    )

    parser.add_argument("input_dir", type=Path, help="Folder containing aligned frames.")
    parser.add_argument("output_dir", type=Path, help="Folder where denoised frames will be saved.")

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.5,
        help="Strength of attenuation. Higher values suppress inconsistent frequencies more. Default: 1.5",
    )

    parser.add_argument(
        "--floor",
        type=float,
        default=0.25,
        help="Minimum retained fraction of any frequency. Range 0-1. Default: 0.25",
    )

    parser.add_argument(
        "--batch-limit",
        type=int,
        default=None,
        help="Optional limit for testing on only the first N frames.",
    )

    parser.add_argument(
    "--mask-sigma",
    type=float,
    default=5.0,
    help="Gaussian smoothing sigma applied to coherence masks."
    )

    parser.add_argument(
        "--radial-cutoff",
        type=float,
        default=0.12,
        help="Radius below which frequencies are protected."
    )

    parser.add_argument(
        "--radial-softness",
        type=float,
        default=0.15,
        help="Transition width for radial frequency gate."
    )

    parser.add_argument(
        "--no-difference",
        action="store_true",
        help="Do not save difference frames."
    )

    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not save mask FITS files."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.floor <= 1.0:
        raise ValueError("--floor must be between 0 and 1.")

    if args.gamma < 0:
        raise ValueError("--gamma must be non-negative.")

    denoise_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gamma=args.gamma,
        floor=args.floor,
        batch_limit=args.batch_limit,
        mask_sigma=args.mask_sigma,
        radial_cutoff=args.radial_cutoff,
        radial_softness=args.radial_softness,
        save_difference=not args.no_difference,
        save_masks=not args.no_masks,
    )


if __name__ == "__main__":
    main()
