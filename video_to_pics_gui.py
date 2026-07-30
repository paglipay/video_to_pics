"""Manual frame capture GUI for videos.

Features:
- Load a folder of videos and switch between files.
- Play/pause and scrub with a timeline to freeze on desired moments.
- Capture frames manually using the existing naming convention:
  <stem>_INSTALL.jpg, <stem>A.jpg, <stem>B.jpg, ..., <stem>.jpg
- Optional filename cleanup via remove-text.
- Optional 4:3 crop with left/center/right horizontal alignment.
- Copies GPS + camera EXIF tags from source video when possible.

Dependencies:
    pip install PySide6 opencv-python piexif
Optional:
    exiftool on PATH for GPS/camera metadata extraction from videos.
"""

from __future__ import annotations

import json
import shutil
import string
import subprocess
from pathlib import Path

import cv2
import piexif
from PySide6.QtCore import QSignalBlocker, QUrl, Qt
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QComboBox,
    QCheckBox,
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".m4v",
    ".wmv",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
}
TARGET_RATIO_4_3 = (4, 3)


def _letter_label(n: int) -> str:
    """Return Excel-style column label for 0-based index n (0->A, 26->AA)."""
    letters = string.ascii_uppercase
    result = ""
    n += 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = letters[remainder] + result
    return result


def _capture_suffix(capture_index: int, planned_count: int) -> str:
    """Existing convention:
    first -> _INSTALL, middle -> A/B/..., last -> base stem (empty suffix).
    """
    if planned_count <= 0:
        raise ValueError("Planned count must be >= 1")
    if capture_index < 0 or capture_index >= planned_count:
        raise ValueError("Capture index outside planned range")

    if planned_count == 1:
        return "_INSTALL"

    last_i = planned_count - 1
    if capture_index == 0:
        return "_INSTALL"
    if capture_index == last_i:
        return ""
    return _letter_label(capture_index - 1)


def _build_output_stem(video_stem: str, remove_text: str | None) -> str:
    if not remove_text:
        return video_stem
    cleaned = video_stem.replace(remove_text, "")
    return cleaned if cleaned else video_stem


def crop_frame_to_ratio(frame, ratio_w: int, ratio_h: int, horizontal_align: str = "center"):
    """Crop a frame to a target ratio. Horizontal alignment applies when cropping width."""
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return frame

    target_ratio = ratio_w / ratio_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        if new_w <= 0 or new_w > w:
            return frame

        if horizontal_align == "left":
            x0 = 0
        elif horizontal_align == "right":
            x0 = w - new_w
        else:
            x0 = (w - new_w) // 2

        x1 = x0 + new_w
        return frame[:, x0:x1]

    if current_ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        if new_h <= 0 or new_h > h:
            return frame

        y0 = (h - new_h) // 2
        y1 = y0 + new_h
        return frame[y0:y1, :]

    return frame


def _read_video_meta(video_path: Path) -> dict:
    """Read GPS + camera tags from a video file via exiftool JSON output."""
    if not shutil.which("exiftool"):
        return {}

    try:
        result = subprocess.run(
            [
                "exiftool",
                "-json",
                "-GPSLatitude",
                "-GPSLatitudeRef",
                "-GPSLongitude",
                "-GPSLongitudeRef",
                "-GPSAltitude",
                "-GPSAltitudeRef",
                "-GPSTimeStamp",
                "-GPSDateStamp",
                "-CreateDate",
                "-Make",
                "-Model",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(result.stdout)
        return data[0] if data else {}
    except Exception:
        return {}


def _parse_gps_value(value: str) -> float | None:
    s = str(value).strip()
    try:
        return abs(float(s.split()[0]))
    except (ValueError, IndexError):
        pass

    try:
        parts = s.replace("deg", "").replace("'", "").replace('"', "").split()
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return d + m / 60 + sec / 3600
    except Exception:
        return None


def _to_rational(value: float, precision: int = 10000) -> tuple[tuple[int, int], ...]:
    deg = int(value)
    minutes = int((value - deg) * 60)
    seconds = round(((value - deg) * 60 - minutes) * 60 * precision)
    return ((deg, 1), (minutes, 1), (seconds, precision))


def _build_gps_ifd(meta: dict) -> dict | None:
    lat_raw = meta.get("GPSLatitude")
    lon_raw = meta.get("GPSLongitude")
    if not lat_raw or not lon_raw:
        return None

    lat = _parse_gps_value(lat_raw)
    lon = _parse_gps_value(lon_raw)
    if lat is None or lon is None:
        return None

    lat_ref = str(meta.get("GPSLatitudeRef", "N"))[0].upper().encode()
    lon_ref = str(meta.get("GPSLongitudeRef", "E"))[0].upper().encode()

    gps_ifd: dict = {
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLatitude: _to_rational(lat),
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
        piexif.GPSIFD.GPSLongitude: _to_rational(lon),
    }

    alt_raw = meta.get("GPSAltitude")
    if alt_raw:
        try:
            alt = float(str(alt_raw).split()[0])
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (round(abs(alt) * 100), 100)
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        except (ValueError, IndexError):
            pass

    return gps_ifd


def _inject_exif(jpg_path: Path, meta: dict) -> None:
    gps_ifd = _build_gps_ifd(meta)
    if not gps_ifd:
        return

    exif_dict: dict = {"GPS": gps_ifd, "0th": {}, "Exif": {}, "1st": {}}

    make = meta.get("Make", "")
    model = meta.get("Model", "")
    if make:
        exif_dict["0th"][piexif.ImageIFD.Make] = str(make).encode()
    if model:
        exif_dict["0th"][piexif.ImageIFD.Model] = str(model).encode()

    try:
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(jpg_path))
    except Exception:
        pass


def _read_frame_at_position(video_path: Path, position_ms: int):
    """Read a single frame using OpenCV at the requested player position (ms)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, position_ms))
        ok, frame = cap.read()
        if ok:
            return frame

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            fallback_idx = max(0, min(frame_count - 1, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
            ok, frame = cap.read()
            if ok:
                return frame

        raise RuntimeError("Could not decode a frame at this position.")
    finally:
        cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video to Pics - Manual Capture")
        self.resize(1280, 780)

        self.video_folder: Path | None = None
        self.video_paths: list[Path] = []
        self.current_video: Path | None = None
        self.current_video_meta: dict = {}
        self.capture_index: int = 0

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget(self)
        self.player.setVideoOutput(self.video_widget)

        self.timeline = QSlider(Qt.Orientation.Horizontal, self)
        self.timeline.setRange(0, 0)
        self.timeline.sliderPressed.connect(self._on_slider_pressed)
        self.timeline.sliderReleased.connect(self._on_slider_released)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)

        self.position_label = QLabel("00:00:00 / 00:00:00")
        self.status_label = QLabel("Load a folder to begin.")

        self.video_list = QListWidget(self)
        self.video_list.currentRowChanged.connect(self._on_video_selected)

        self.btn_load_folder = QPushButton("Load Folder")
        self.btn_load_folder.clicked.connect(self._choose_folder)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._play)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._pause)

        self.btn_prev5 = QPushButton("-5s")
        self.btn_prev5.clicked.connect(lambda: self._seek_relative(-5000))

        self.btn_next5 = QPushButton("+5s")
        self.btn_next5.clicked.connect(lambda: self._seek_relative(5000))

        self.capture_count_spin = QSpinBox(self)
        self.capture_count_spin.setRange(1, 500)
        self.capture_count_spin.setValue(8)
        self.capture_count_spin.valueChanged.connect(self._update_next_name_preview)

        self.remove_text_input = QLineEdit(self)
        self.remove_text_input.setPlaceholderText("Optional substring to remove from video stem")
        self.remove_text_input.textChanged.connect(self._update_next_name_preview)

        self.crop_4_3_checkbox = QCheckBox("Crop capture to 4:3")

        self.crop_align_combo = QComboBox(self)
        self.crop_align_combo.addItems(["center", "left", "right"])

        self.output_dir_input = QLineEdit(self)
        self.output_dir_input.setPlaceholderText("Choose output folder")

        self.btn_output_dir = QPushButton("Choose Output")
        self.btn_output_dir.clicked.connect(self._choose_output_dir)

        self.btn_capture = QPushButton("Capture Next")
        self.btn_capture.clicked.connect(self._capture_current_frame)

        self.btn_reset_series = QPushButton("Reset Series")
        self.btn_reset_series.clicked.connect(self._reset_capture_series)

        self.next_name_label = QLabel("Next file: -")

        controls_left = QVBoxLayout()
        controls_left.addWidget(self.btn_load_folder)
        controls_left.addWidget(QLabel("Videos in folder:"))
        controls_left.addWidget(self.video_list, 1)

        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout()

        playback_buttons = QHBoxLayout()
        playback_buttons.addWidget(self.btn_prev5)
        playback_buttons.addWidget(self.btn_play)
        playback_buttons.addWidget(self.btn_pause)
        playback_buttons.addWidget(self.btn_next5)
        playback_layout.addLayout(playback_buttons)
        playback_layout.addWidget(self.timeline)
        playback_layout.addWidget(self.position_label)
        playback_group.setLayout(playback_layout)

        capture_group = QGroupBox("Capture Settings")
        capture_form = QFormLayout()
        capture_form.addRow("Planned captures", self.capture_count_spin)
        capture_form.addRow("Remove text", self.remove_text_input)
        capture_form.addRow("Crop mode", self.crop_4_3_checkbox)
        capture_form.addRow("Crop align", self.crop_align_combo)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_input, 1)
        output_row.addWidget(self.btn_output_dir)
        output_container = QWidget()
        output_container.setLayout(output_row)
        capture_form.addRow("Output folder", output_container)

        capture_group.setLayout(capture_form)

        capture_buttons = QHBoxLayout()
        capture_buttons.addWidget(self.btn_capture)
        capture_buttons.addWidget(self.btn_reset_series)

        right_panel = QVBoxLayout()
        right_panel.addWidget(playback_group)
        right_panel.addWidget(capture_group)
        right_panel.addLayout(capture_buttons)
        right_panel.addWidget(self.next_name_label)
        right_panel.addWidget(self.status_label)
        right_panel.addStretch(1)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.addLayout(controls_left, 1)

        video_and_controls = QVBoxLayout()
        video_and_controls.addWidget(self.video_widget, 3)
        video_and_controls.addLayout(right_panel, 2)

        root_layout.addLayout(video_and_controls, 3)
        self.setCentralWidget(root)

        self._update_next_name_preview()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose folder with videos")
        if not folder:
            return

        self.video_folder = Path(folder)
        self.video_paths = sorted(
            p for p in self.video_folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

        self.video_list.clear()
        for path in self.video_paths:
            self.video_list.addItem(path.name)

        if not self.video_paths:
            self.status_label.setText("No supported video files found in this folder.")
            self.current_video = None
            self._update_next_name_preview()
            return

        self.video_list.setCurrentRow(0)
        self.status_label.setText(f"Loaded {len(self.video_paths)} video(s) from folder.")

    def _on_video_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.video_paths):
            return

        video_path = self.video_paths[row]
        self.current_video = video_path
        self.current_video_meta = _read_video_meta(video_path)
        self.capture_index = 0

        if not self.output_dir_input.text().strip():
            default_output = video_path.parent / f"{video_path.stem}_frames"
            self.output_dir_input.setText(str(default_output))

        self.player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.player.pause()

        exif_status = "EXIF/GPS ready" if self.current_video_meta else "No EXIF/GPS metadata"
        self.status_label.setText(f"Loaded: {video_path.name} | {exif_status}")
        self._update_next_name_preview()

    def _choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if folder:
            self.output_dir_input.setText(folder)
            self._update_next_name_preview()

    def _play(self) -> None:
        self.player.play()

    def _pause(self) -> None:
        self.player.pause()

    def _seek_relative(self, delta_ms: int) -> None:
        target = max(0, min(self.player.duration(), self.player.position() + delta_ms))
        self.player.setPosition(target)

    def _on_duration_changed(self, duration_ms: int) -> None:
        self.timeline.setRange(0, max(0, duration_ms))
        self._update_position_label(self.player.position(), duration_ms)

    def _on_position_changed(self, position_ms: int) -> None:
        with QSignalBlocker(self.timeline):
            self.timeline.setValue(position_ms)
        self._update_position_label(position_ms, self.player.duration())

    def _on_slider_pressed(self) -> None:
        self.player.pause()

    def _on_slider_released(self) -> None:
        self.player.setPosition(self.timeline.value())

    def _format_ms(self, ms: int) -> str:
        total_s = max(0, ms // 1000)
        h = total_s // 3600
        m = (total_s % 3600) // 60
        s = total_s % 60
        return f"{h:02}:{m:02}:{s:02}"

    def _update_position_label(self, pos_ms: int, dur_ms: int) -> None:
        self.position_label.setText(f"{self._format_ms(pos_ms)} / {self._format_ms(dur_ms)}")

    def _next_output_name(self) -> str:
        if not self.current_video:
            return "-"

        planned = self.capture_count_spin.value()
        if self.capture_index >= planned:
            return "(capture series complete - reset to capture again)"

        stem = _build_output_stem(self.current_video.stem, self.remove_text_input.text().strip())
        suffix = _capture_suffix(self.capture_index, planned)
        return f"{stem}{suffix}.jpg"

    def _update_next_name_preview(self) -> None:
        self.next_name_label.setText(f"Next file: {self._next_output_name()}")

    def _reset_capture_series(self) -> None:
        self.capture_index = 0
        self._update_next_name_preview()
        self.status_label.setText("Capture series reset to first name (_INSTALL).")

    def _capture_current_frame(self) -> None:
        if not self.current_video:
            QMessageBox.warning(self, "No video", "Load a video first.")
            return

        out_dir_text = self.output_dir_input.text().strip()
        if not out_dir_text:
            QMessageBox.warning(self, "No output folder", "Choose an output folder first.")
            return

        planned = self.capture_count_spin.value()
        if self.capture_index >= planned:
            QMessageBox.information(
                self,
                "Series complete",
                "Planned captures are complete. Click 'Reset Series' to start over.",
            )
            return

        output_dir = Path(out_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = _build_output_stem(self.current_video.stem, self.remove_text_input.text().strip())
        suffix = _capture_suffix(self.capture_index, planned)
        out_path = output_dir / f"{stem}{suffix}.jpg"

        try:
            position_ms = int(self.player.position())
            frame = _read_frame_at_position(self.current_video, position_ms)

            if self.crop_4_3_checkbox.isChecked():
                frame = crop_frame_to_ratio(
                    frame,
                    ratio_w=TARGET_RATIO_4_3[0],
                    ratio_h=TARGET_RATIO_4_3[1],
                    horizontal_align=self.crop_align_combo.currentText(),
                )

            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise RuntimeError("OpenCV failed to write the output image.")

            if self.current_video_meta:
                _inject_exif(out_path, self.current_video_meta)

            self.capture_index += 1
            self._update_next_name_preview()
            self.status_label.setText(f"Saved: {out_path.name}")

        except Exception as exc:
            QMessageBox.critical(self, "Capture failed", str(exc))


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
