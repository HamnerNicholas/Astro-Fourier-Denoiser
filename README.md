# Astro Fourier Denoiser

An experimental astrophotography denoising application that uses Fourier-domain coherence analysis across multiple subframes to reduce noise while preserving real astronomical signal.

Unlike traditional single-image denoisers, Astro Fourier Denoiser compares frequency content across aligned light frames and attenuates frequency components that are inconsistent between exposures.

---

## Features

### Image Processing

- Automatic FITS loading
- Bayer pattern detection from FITS headers
- Automatic debayering (RGGB, BGGR, GRBG, GBRG)
- Subframe alignment using phase correlation
- Multi-frame Fourier coherence analysis
- Frequency-domain denoising
- Gaussian mask smoothing
- Radial frequency protection
- Difference frame generation
- Per-channel RGB denoising

### Diagnostics

- MAD (Median Absolute Deviation) metrics
- Background MAD estimation
- Difference frame export
- Frequency mask export
- Noise reduction reporting

### Desktop Application

- PyQt6 desktop interface
- Folder-based workflow
- Live processing log
- Progress tracking
  - Debayering
  - Alignment
  - Denoising
- Raw light stack preview
- Denoised stack preview
- Side-by-side comparison

---

## How It Works

The denoiser uses a frequency-domain coherence metric.

### Processing Pipeline

```text
Light Frames
     ↓
Debayer
     ↓
Alignment
     ↓
Fourier Transform
     ↓
Coherence Analysis
     ↓
Gaussian Smoothing
     ↓
Radial Frequency Gate
     ↓
Inverse Transform
     ↓
Denoised Frames
```

The goal is to attenuate frequencies that appear inconsistent across exposures while preserving frequencies associated with real astronomical signal.

### Coherence Estimation

For each frequency component:

```
Coherence = |mean(F)| / mean(|F|)
```

Frequencies that remain stable across multiple aligned exposures are preserved, while inconsistent frequencies are attenuated.

---

## Supported Formats

### Input

- FIT
- FITS
- FTS
- TIFF
- PNG
- JPEG

### Output

- FITS
- TIFF
- PNG
- Difference Frames
- Coherence Masks

---

## Example Results

Current testing on M101 datasets shows:

| Metric | Result |
|----------|----------|
| Average MAD Reduction | ~7–8% |
| Red Channel Noise Reduction | ~17% |
| Green Channel Noise Reduction | ~8% |
| Blue Channel | Currently under investigation |
| Visible Artifact Reduction | Significant |

Results vary by target, integration time, sky conditions, and acquisition setup.

---

## Installation

### Requirements

```bash
pip install numpy
pip install scipy
pip install astropy
pip install imageio
pip install scikit-image
pip install opencv-python
pip install pillow
pip install pyqt6
```

---

## Running the GUI

```bash
python AstroDenoiserApp.py
```

---

## Running the Engine

```bash
python DenoiseEngine.py input_folder output_folder
```

Example:

```bash
python DenoiseEngine.py lights denoised ^
    --gamma 1.0 ^
    --floor 0.35 ^
    --mask-sigma 5.0 ^
    --radial-cutoff 0.12 ^
    --radial-softness 0.15
```

---

## Parameters

### Gamma

Controls coherence attenuation strength.

Higher values:

- More aggressive noise reduction
- Increased risk of signal suppression

Lower values:

- More conservative denoising
- Better detail preservation

---

### Floor

Minimum retained frequency amplitude.

Higher values:

- Preserves more detail
- Less aggressive denoising

Lower values:

- Stronger denoising
- Greater risk of over-processing

---

### Mask Sigma

Gaussian smoothing applied to coherence masks.

Higher values:

- Smoother masks
- Reduced FFT artifacts
- More conservative filtering

Lower values:

- More localized frequency suppression
- Increased risk of mask imprinting

---

### Radial Cutoff

Protects low-frequency astronomical structures such as:

- Galaxies
- Nebulae
- Large-scale gradients

---

### Radial Softness

Controls how gradually the radial gate transitions from protected frequencies to denoised frequencies.

---

## GUI Features

### Raw Stack Preview

Generates a quick median stack from loaded light frames.

Purpose:

- Verify correct dataset selection
- Confirm framing
- Check image quality before processing

### Denoised Stack Preview

Automatically generated after denoising completes.

Purpose:

- Immediate visual comparison
- Evaluate denoising effectiveness
- Compare against raw stack

### Progress Tracking

Separate progress bars for:

- Debayering
- Alignment
- Denoising

### Metrics Summary

The application reports:

- Average MAD reduction
- Average background MAD reduction

These metrics are calculated directly from the processed light frames.

---

## Difference Frames

Difference frames are generated using:

```text
Original Frame
      -
Denoised Frame
```

These images help determine whether the algorithm is removing:

- Random noise
- Fixed pattern noise
- Real astronomical signal

A successful result should show primarily noise and sensor artifacts rather than stars or galaxy structure.

---

## Coherence Masks

The application can export:

- Red mask
- Green mask
- Blue mask
- Luminance mask

These masks can be inspected in Siril or other FITS viewers to understand which frequencies are being attenuated.

---

## Current Status

This project is currently experimental.

The algorithm is under active development and should not yet be considered a replacement for established astrophotography workflows such as:

- PixInsight
- Siril
- AstroPixelProcessor
- DeepSkyStacker

Instead, Astro Fourier Denoiser serves as a research platform for exploring frequency-domain coherence denoising techniques.

---

## Roadmap

### Planned Features

- GPU acceleration using CuPy
- Integrated stacking
- Automatic benchmarking
- Live frequency mask viewer
- Star FWHM analysis
- Noise reduction scoring
- Parameter presets
- Run history database
- Batch parameter optimization

### Long-Term Goals

- Standalone desktop application
- One-click processing pipeline
- Project management system
- Advanced visualization tools
- Multi-session datasets

---

## Known Limitations

- Blue-channel denoising remains under active investigation
- Requires aligned light frames for best results
- Current implementation focuses on RGB OSC data
- Limited testing across different targets and acquisition systems

---

## Contributing

Contributions, testing, bug reports, and algorithm suggestions are welcome.

Areas of particular interest:

- Signal preservation metrics
- Alternative coherence models
- GPU acceleration
- Advanced registration techniques
- Benchmark datasets

---

## License

MIT License

---

## Author

Nicholas Hamner
