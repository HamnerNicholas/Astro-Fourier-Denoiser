#!/usr/bin/env python3
"""
Astro Fourier Denoiser - Desktop Front End Prototype

This GUI wraps your existing DenoiseEngine.py command-line script.

Install:
    pip install PyQt6

Run:
    python AstroDenoiserApp.py

Expected folder layout:
    DenoisingApp/
    ├── DenoiseEngine.py
    └── AstroDenoiserApp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
    QMessageBox,
    QCheckBox,
)

import re
from PyQt6.QtWidgets import QProgressBar

import numpy as np
from astropy.io import fits
from PIL import Image
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout

import cv2
import csv

def debayer_frame(frame: np.ndarray, pattern="RGGB") -> np.ndarray:
    frame = frame.astype(np.uint16)

    patterns = {
        "RGGB": cv2.COLOR_BAYER_RG2RGB,
        "BGGR": cv2.COLOR_BAYER_BG2RGB,
        "GRBG": cv2.COLOR_BAYER_GR2RGB,
        "GBRG": cv2.COLOR_BAYER_GB2RGB,
    }

    return cv2.cvtColor(frame, patterns[pattern]).astype(np.float32)

class AstroDenoiserApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Astro Fourier Denoiser")
        self.resize(900, 600)

        self.process: QProcess | None = None

        self.engine_path = Path(__file__).with_name("DenoiseEngine.py")

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.0, 5.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)

        self.floor_spin = QDoubleSpinBox()
        self.floor_spin.setRange(0.0, 1.0)
        self.floor_spin.setSingleStep(0.01)
        self.floor_spin.setValue(0.35)

        self.mask_sigma_spin = QDoubleSpinBox()
        self.mask_sigma_spin.setRange(0.0, 20.0)
        self.mask_sigma_spin.setSingleStep(0.5)
        self.mask_sigma_spin.setValue(5.0)

        self.radial_cutoff_spin = QDoubleSpinBox()
        self.radial_cutoff_spin.setRange(0.0, 0.5)
        self.radial_cutoff_spin.setSingleStep(0.01)
        self.radial_cutoff_spin.setValue(0.12)

        self.radial_softness_spin = QDoubleSpinBox()
        self.radial_softness_spin.setRange(0.01, 0.5)
        self.radial_softness_spin.setSingleStep(0.01)
        self.radial_softness_spin.setValue(0.15)

        self.batch_limit_spin = QSpinBox()
        self.batch_limit_spin.setRange(0, 10000)
        self.batch_limit_spin.setValue(14)
        self.batch_limit_spin.setSpecialValueText("All")

        self.save_difference_check = QCheckBox("Save difference frames")
        self.save_difference_check.setChecked(True)

        self.save_masks_check = QCheckBox("Save masks")
        self.save_masks_check.setChecked(True)

        self.start_button = QPushButton("Start Denoising")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        self.preview_button = QPushButton("Preview Raw Light Stack")

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)

        self.debayer_progress = QProgressBar()
        self.align_progress = QProgressBar()
        self.denoise_progress = QProgressBar()

        self.raw_title = QLabel("Raw Stack Preview")
        self.denoised_title = QLabel("Denoised Stack Preview")

        for label in [self.raw_title, self.denoised_title]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for bar in [
            self.debayer_progress,
            self.align_progress,
            self.denoise_progress,
        ]:
            bar.setMinimum(0)
            bar.setMaximum(100)
            bar.setValue(0)

        self.raw_preview_label = QLabel("")
        self.denoised_preview_label = QLabel("")

        for label in [self.raw_preview_label, self.denoised_preview_label]:
            label.setMinimumSize(350, 350)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setScaledContents(False)


        for label in [self.raw_preview_label, self.denoised_preview_label]:
            label.setMinimumSize(400, 400)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setScaledContents(False)
            label.setStyleSheet("border: 1px solid #777; background-color: #222; color: #ccc;")

        self.raw_preview_pixmap = None
        self.denoised_preview_pixmap = None

        self.raw_stats_label = QLabel("Noise: —")
        self.denoised_stats_label = QLabel("Noise: —")

        for label in [self.raw_stats_label, self.denoised_stats_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QGridLayout()
        central.setLayout(layout)

        input_button = QPushButton("Browse...")
        input_button.clicked.connect(self.choose_input_folder)

        output_button = QPushButton("Browse...")
        output_button.clicked.connect(self.choose_output_folder)

        layout.addWidget(QLabel("Input folder:"), 0, 0)
        layout.addWidget(self.input_edit, 0, 1)
        layout.addWidget(input_button, 0, 2)

        layout.addWidget(QLabel("Output folder:"), 1, 0)
        layout.addWidget(self.output_edit, 1, 1)
        layout.addWidget(output_button, 1, 2)

        layout.addWidget(QLabel("Gamma:"), 2, 0)
        layout.addWidget(self.gamma_spin, 2, 1)

        layout.addWidget(QLabel("Floor:"), 3, 0)
        layout.addWidget(self.floor_spin, 3, 1)

        layout.addWidget(QLabel("Mask smoothing sigma:"), 4, 0)
        layout.addWidget(self.mask_sigma_spin, 4, 1)

        layout.addWidget(QLabel("Radial cutoff:"), 5, 0)
        layout.addWidget(self.radial_cutoff_spin, 5, 1)

        layout.addWidget(QLabel("Radial softness:"), 6, 0)
        layout.addWidget(self.radial_softness_spin, 6, 1)

        layout.addWidget(QLabel("Batch limit:"), 7, 0)
        layout.addWidget(self.batch_limit_spin, 7, 1)

        layout.addWidget(self.save_difference_check, 8, 1)
        layout.addWidget(self.save_masks_check, 9, 1)

        preview_column = QHBoxLayout()

        # Left side
        raw_layout = QVBoxLayout()
        raw_layout.addWidget(self.raw_title)
        raw_layout.addWidget(self.raw_preview_label)
        raw_layout.addWidget(self.raw_title)
        raw_layout.addWidget(self.raw_preview_label)
        raw_layout.addWidget(self.raw_stats_label)

        # Right side
        denoised_layout = QVBoxLayout()
        denoised_layout.addWidget(self.denoised_title)
        denoised_layout.addWidget(self.denoised_preview_label)
        denoised_layout.addWidget(self.denoised_title)
        denoised_layout.addWidget(self.denoised_preview_label)
        denoised_layout.addWidget(self.denoised_stats_label)

        preview_column.addLayout(raw_layout, 1)
        preview_column.addLayout(denoised_layout, 1)

        layout.addLayout(preview_column, 16, 0, 1, 3)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.preview_button)


        layout.addLayout(button_row, 10, 1)

        layout.addWidget(QLabel("Debayering:"), 11, 0)
        layout.addWidget(self.debayer_progress, 11, 1, 1, 2)

        layout.addWidget(QLabel("Aligning:"), 12, 0)
        layout.addWidget(self.align_progress, 12, 1, 1, 2)

        layout.addWidget(QLabel("Denoising:"), 13, 0)
        layout.addWidget(self.denoise_progress, 13, 1, 1, 2)

        layout.addWidget(QLabel("Log:"), 14, 0)
        layout.addWidget(self.log_box, 15, 0, 1, 3)

        self.start_button.clicked.connect(self.start_denoising)
        self.stop_button.clicked.connect(self.stop_denoising)
        self.preview_button.clicked.connect(self.preview_raw_stack)

    def choose_input_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose input folder")
        if folder:
            self.input_edit.setText(folder)

    def choose_output_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.output_edit.setText(folder)
    def update_progress_from_text(self, text: str) -> None:
        # Total frames
        found_match = re.search(r"Found\s+(\d+)\s+frame", text)
        if found_match:
            total = int(found_match.group(1))

            for bar in [
                self.debayer_progress,
                self.align_progress,
                self.denoise_progress,
            ]:
                bar.setMaximum(total)
                bar.setValue(0)

            return

        # Debayering lines
        if "Debayering " in text:
            self.debayer_progress.setValue(
                min(
                    self.debayer_progress.value() + 1,
                    self.debayer_progress.maximum(),
                )
            )
            return

        # Alignment lines: Frame 2, Frame 3, ...
        align_match = re.search(r"Frame\s+(\d+):\s+dy=", text)
        if align_match:
            current_frame = int(align_match.group(1))
            self.align_progress.setValue(current_frame)
            return

        # Denoising lines: [7/14] saved ...
        denoise_match = re.search(r"\[(\d+)/(\d+)\]", text)
        if denoise_match:
            current = int(denoise_match.group(1))
            total = int(denoise_match.group(2))

            self.denoise_progress.setMaximum(total)
            self.denoise_progress.setValue(current)
            return
        
    def append_log(self, text: str) -> None:
        self.log_box.appendPlainText(text.rstrip())
        self.update_progress_from_text(text)

    def validate_inputs(self) -> bool:
        input_dir = Path(self.input_edit.text().strip())
        output_dir = Path(self.output_edit.text().strip())

        if not self.engine_path.exists():
            QMessageBox.critical(
                self,
                "Missing engine",
                f"Could not find DenoiseEngine.py at:\n{self.engine_path}",
            )
            return False

        if not input_dir.exists():
            QMessageBox.critical(self, "Invalid input", "Input folder does not exist.")
            return False

        if not output_dir:
            QMessageBox.critical(self, "Invalid output", "Choose an output folder.")
            return False

        return True

    def start_denoising(self) -> None:
        if not self.validate_inputs():
            return

        input_dir = Path(self.input_edit.text().strip())
        output_dir = Path(self.output_edit.text().strip())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for bar in [
            self.debayer_progress,
            self.align_progress,
            self.denoise_progress,
        ]:
            bar.setValue(0)
            bar.setMaximum(100)

        args = [
            "-u",
            str(self.engine_path),
            str(input_dir),
            str(output_dir),
            "--gamma",
            str(self.gamma_spin.value()),
            "--floor",
            str(self.floor_spin.value()),
            "--mask-sigma",
            str(self.mask_sigma_spin.value()),
            "--radial-cutoff",
            str(self.radial_cutoff_spin.value()),
            "--radial-softness",
            str(self.radial_softness_spin.value()),
        ]

        if self.batch_limit_spin.value() > 0:
            args.extend(["--batch-limit", str(self.batch_limit_spin.value())])

        # These options require matching CLI flags in DenoiseEngine.py.
        # If your engine does not support them yet, either add the flags
        # or remove this section.
        if not self.save_difference_check.isChecked():
            args.append("--no-difference")

        if not self.save_masks_check.isChecked():
            args.append("--no-masks")

        self.log_box.clear()
        self.append_log("Starting denoise run...")
        self.append_log("Command:")
        self.append_log(f"{sys.executable} " + " ".join(args))

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(args)

        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.process.start()

    def read_stdout(self) -> None:
        if self.process is None:
            return

        data = self.process.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self.append_log(data)

    def read_stderr(self) -> None:
        if self.process is None:
            return

        data = self.process.readAllStandardError().data().decode(errors="replace")
        if data:
            self.append_log(data)

    def process_finished(self, exit_code: int, exit_status) -> None:
        if exit_code == 0:
            for bar in [
                self.debayer_progress,
                self.align_progress,
                self.denoise_progress,
            ]:
                bar.setValue(bar.maximum())

            self.preview_denoised_stack()

        self.append_log("")
        self.append_log(f"Process finished with exit code {exit_code}.")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.process = None

    def stop_denoising(self) -> None:
        if self.process is not None:
            self.append_log("Stopping process...")
            self.process.kill()

    def preview_raw_stack(self) -> None:
        input_dir = Path(self.input_edit.text().strip())

        files = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in {".fit", ".fits", ".fts"}
        )

        if not files:
            QMessageBox.warning(self, "No FITS files", "No FITS lights found.")
            return

        self.append_log(f"Building raw preview stack from {len(files)} lights...")

        stack = self.quick_median_stack(files, max_frames=20)
        raw_stats = self.estimate_background_noise(stack)
        self.raw_stats_label.setText(self.format_noise_stats(raw_stats))
        preview = self.stretch_for_preview(stack)

        preview_path = Path(self.output_edit.text().strip()) / "raw_preview_stack.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(preview).save(preview_path)

        self.raw_preview_pixmap = QPixmap(str(preview_path))
        self.update_preview_pixmaps()
        self.append_log(f"Preview saved: {preview_path}")


    def quick_median_stack(self, files, max_frames=20):

        frames = []

        for path in files[:max_frames]:
            data = fits.getdata(path).astype(np.float32)
            data = np.squeeze(data)

            header = fits.getheader(path)
            pattern = header.get("BAYERPAT")

            if pattern and data.ndim == 2:
                data = debayer_frame(data, pattern)

            if data.ndim == 3 and data.shape[0] == 3:
                data = np.moveaxis(data, 0, -1)

            frames.append(data)

        return np.median(np.stack(frames, axis=0), axis=0)


    def stretch_for_preview(self, img):
        img = img.astype(np.float32)

        if img.ndim == 3 and img.shape[-1] == 3:
            channels = []
            for ch in range(3):
                c = img[..., ch]
                lo, hi = np.percentile(c, [0.5, 99.5])
                c = np.clip((c - lo) / (hi - lo + 1e-8), 0, 1)
                channels.append(c)
            img = np.stack(channels, axis=-1)
        else:
            lo, hi = np.percentile(img, [0.5, 99.5])
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

        return (img * 255).astype(np.uint8)
    
    def update_preview_pixmaps(self) -> None:
        pairs = [
            (self.raw_preview_pixmap, self.raw_preview_label),
            (self.denoised_preview_pixmap, self.denoised_preview_label),
        ]

        for pixmap, label in pairs:
            if pixmap is None:
                continue

            scaled = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

            label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_preview_pixmaps()
    
    def preview_denoised_stack(self) -> None:
        output_dir = Path(self.output_edit.text().strip())

        files = sorted(
            p for p in output_dir.iterdir()
            if (
                p.suffix.lower() in {".fit", ".fits", ".fts"}
                and "_denoised" in p.stem.lower()
            )
        )

        if not files:
            self.append_log("No denoised FITS files found for preview.")
            return

        self.append_log(f"Building denoised preview stack from {len(files)} frames...")

        stack = self.quick_median_stack(files, max_frames=20)
        preview = self.stretch_for_preview(stack)

        summary = self.load_denoise_metrics_summary()
        self.denoised_stats_label.setText(summary)

        preview_path = output_dir / "denoised_preview_stack.png"
        Image.fromarray(preview).save(preview_path)

        self.denoised_preview_pixmap = QPixmap(str(preview_path))
        self.update_preview_pixmaps()

        self.append_log(f"Denoised preview saved: {preview_path}")

    def estimate_background_noise(self, img: np.ndarray) -> dict:
        img = img.astype(np.float32)

        if img.ndim == 2:
            cutoff = np.percentile(img, 80)
            bg = img[img <= cutoff]
            return {
                "mono": float(np.std(bg))
            }

        results = {}

        for ch, name in enumerate(["R", "G", "B"]):
            channel = img[..., ch]
            cutoff = np.percentile(channel, 80)
            bg = channel[channel <= cutoff]
            results[name] = float(np.std(bg))

        return results

    def format_noise_stats(self, stats: dict) -> str:
        if "mono" in stats:
            return f"Noise σ: {stats['mono']:.3f}"

        return (
            f"Noise σ  "
            f"R: {stats['R']:.3f}   "
            f"G: {stats['G']:.3f}   "
            f"B: {stats['B']:.3f}"
        )
    
    def load_denoise_metrics_summary(self) -> str:
        output_dir = Path(self.output_edit.text().strip())
        metrics_path = output_dir / "denoise_metrics.csv"

        if not metrics_path.exists():
            return "No metrics CSV found."

        mad_reductions = []
        background_reductions = []

        with metrics_path.open("r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row.get("mad_reduction_pct"):
                    mad_reductions.append(float(row["mad_reduction_pct"]))

                if row.get("background_mad_reduction_pct"):
                    background_reductions.append(
                        float(row["background_mad_reduction_pct"])
                    )

        avg_mad = sum(mad_reductions) / len(mad_reductions)
        avg_bg = sum(background_reductions) / len(background_reductions)

        return (
            f"Avg MAD reduction: {avg_mad:.2f}%   "
            f"Avg background MAD reduction: {avg_bg:.2f}%"
        )



def main() -> None:
    app = QApplication(sys.argv)
    window = AstroDenoiserApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
