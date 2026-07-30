"""PySide6 GUI wrapper for image_cleaner.py with manual touch-up support.

This app preserves the existing image_cleaner.py workflow and options, and adds
an optional manual mode to redact objects missed by YOLO detections.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QProcess, Qt, QRect, QPoint
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


DEFAULT_SCRIPT_PATH = Path(r"C:\Users\pagli\Documents\Projects\lausd_tools\image_cleaner.py")

EXIF_ORIENTATION = 0x0112
EXIF_USER_COMMENT = 0x9286
EXIF_GPS_IFD = 0x8825
CLEANED_MARKER = b"image_cleaner:v1"


def build_output_exif(src_exif_bytes: bytes | None, strip_gps: bool) -> bytes | None:
    exif = Image.Exif()
    if src_exif_bytes:
        try:
            exif.load(src_exif_bytes)
        except Exception:
            exif = Image.Exif()

    exif[EXIF_ORIENTATION] = 1
    exif[EXIF_USER_COMMENT] = b"ASCII\x00\x00\x00" + CLEANED_MARKER

    if strip_gps:
        try:
            gps = exif.get_ifd(EXIF_GPS_IFD)
            if gps:
                gps.clear()
        except Exception:
            pass
        try:
            if EXIF_GPS_IFD in exif:
                del exif[EXIF_GPS_IFD]
        except Exception:
            pass

    try:
        return exif.tobytes()
    except Exception:
        return src_exif_bytes


class RectCanvas(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(700, 450)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #181818; color: #cccccc;")
        self.setText("Load an image folder and select an image")

        self.image_np: np.ndarray | None = None
        self.rects: list[tuple[int, int, int, int]] = []
        self.dragging = False
        self.start_pos: QPoint | None = None
        self.current_pos: QPoint | None = None

    def set_image(self, image_np: np.ndarray | None) -> None:
        self.image_np = image_np
        self.rects.clear()
        self.start_pos = None
        self.current_pos = None
        self.dragging = False
        self.update()

    def clear_rects(self) -> None:
        self.rects.clear()
        self.update()

    def undo_rect(self) -> None:
        if self.rects:
            self.rects.pop()
            self.update()

    def get_rects(self) -> list[tuple[int, int, int, int]]:
        return list(self.rects)

    def _draw_rect(self, painter: QPainter, rect_img: tuple[int, int, int, int], scale: float, x_off: int, y_off: int) -> None:
        x, y, w, h = rect_img
        painter.drawRect(
            QRect(
                int(x * scale + x_off),
                int(y * scale + y_off),
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            )
        )

    def _map_widget_to_image(self, pt: QPoint, img_w: int, img_h: int) -> tuple[int, int] | None:
        scale = min(self.width() / img_w, self.height() / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        x_off = (self.width() - disp_w) // 2
        y_off = (self.height() - disp_h) // 2

        x = pt.x() - x_off
        y = pt.y() - y_off
        if x < 0 or y < 0 or x >= disp_w or y >= disp_h:
            return None

        ix = int(x / scale)
        iy = int(y / scale)
        ix = max(0, min(img_w - 1, ix))
        iy = max(0, min(img_h - 1, iy))
        return ix, iy

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.image_np is None:
            return
        self.dragging = True
        self.start_pos = event.position().toPoint()
        self.current_pos = self.start_pos
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.dragging:
            self.current_pos = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.dragging or self.image_np is None or self.start_pos is None:
            return
        self.dragging = False
        end_pos = event.position().toPoint()
        img_h, img_w = self.image_np.shape[:2]
        p1 = self._map_widget_to_image(self.start_pos, img_w, img_h)
        p2 = self._map_widget_to_image(end_pos, img_w, img_h)
        self.start_pos = None
        self.current_pos = None
        if p1 is None or p2 is None:
            self.update()
            return

        x1, y1 = p1
        x2, y2 = p2
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w > 2 and h > 2:
            self.rects.append((x, y, w, h))
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.image_np is None:
            return

        rgb = cv2.cvtColor(self.image_np, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]
        qimg = QImage(rgb.data, img_w, img_h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        scale = min(self.width() / img_w, self.height() / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        x_off = (self.width() - disp_w) // 2
        y_off = (self.height() - disp_h) // 2

        painter = QPainter(self)
        painter.drawPixmap(x_off, y_off, disp_w, disp_h, pix)

        pen = QPen(Qt.GlobalColor.red, 2)
        painter.setPen(pen)
        for rect_img in self.rects:
            self._draw_rect(painter, rect_img, scale, x_off, y_off)

        if self.dragging and self.start_pos and self.current_pos:
            painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
            painter.drawRect(QRect(self.start_pos, self.current_pos).normalized())

        painter.end()


class ManualTouchupDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, strip_gps_default: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manual Touch-up for Missed Objects")
        self.resize(1200, 820)

        self.folder_path: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index: int = -1

        self.current_image_bgr: np.ndarray | None = None
        self.src_exif_bytes: bytes | None = None
        self.src_icc_profile: bytes | None = None
        self.src_quantization = None
        self.current_ext = ".jpg"

        self.canvas = RectCanvas()

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Folder with images to manually touch up")
        self.btn_browse_folder = QPushButton("Browse")
        self.btn_browse_folder.clicked.connect(self._choose_folder)

        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._on_image_selected)

        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["pixelate", "blur", "inpaint"])

        self.blur_sigma_spin = QSpinBox()
        self.blur_sigma_spin.setRange(1, 500)
        self.blur_sigma_spin.setValue(45)

        self.inpaint_radius_spin = QSpinBox()
        self.inpaint_radius_spin.setRange(1, 500)
        self.inpaint_radius_spin.setValue(15)

        self.pixelate_block_spin = QSpinBox()
        self.pixelate_block_spin.setRange(1, 500)
        self.pixelate_block_spin.setValue(20)

        self.strip_gps_cb = QCheckBox("Strip GPS metadata on save")
        self.strip_gps_cb.setChecked(strip_gps_default)

        self.btn_prev = QPushButton("Prev")
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self._next_image)

        self.btn_undo_box = QPushButton("Undo Box")
        self.btn_undo_box.clicked.connect(self.canvas.undo_rect)
        self.btn_clear_boxes = QPushButton("Clear Boxes")
        self.btn_clear_boxes.clicked.connect(self.canvas.clear_rects)

        self.btn_apply = QPushButton("Apply Effect to Boxes")
        self.btn_apply.clicked.connect(self._apply_effect)

        self.btn_save = QPushButton("Save (Overwrite)")
        self.btn_save.clicked.connect(self._save_current)

        self.status_label = QLabel("Draw boxes around missed objects, then apply effect and save.")

        self._build_ui()

    def _build_ui(self) -> None:
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_input, 1)
        folder_row.addWidget(self.btn_browse_folder)

        nav_row = QHBoxLayout()
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.image_combo, 1)
        nav_row.addWidget(self.btn_next)

        controls = QFormLayout()
        controls.addRow("Folder", self._wrap(folder_row))
        controls.addRow("Image", self._wrap(nav_row))
        controls.addRow("Effect", self.effect_combo)
        controls.addRow("Blur sigma", self.blur_sigma_spin)
        controls.addRow("Inpaint radius", self.inpaint_radius_spin)
        controls.addRow("Pixelate block", self.pixelate_block_spin)
        controls.addRow("", self.strip_gps_cb)

        top = QGroupBox("Manual Redaction")
        top.setLayout(controls)

        action_row = QHBoxLayout()
        action_row.addWidget(self.btn_undo_box)
        action_row.addWidget(self.btn_clear_boxes)
        action_row.addWidget(self.btn_apply)
        action_row.addWidget(self.btn_save)

        root_layout = QVBoxLayout()
        root_layout.addWidget(top)
        root_layout.addLayout(action_row)
        root_layout.addWidget(self.canvas, 1)
        root_layout.addWidget(self.status_label)
        self.setLayout(root_layout)

    def _wrap(self, layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if not folder:
            return
        self.folder_input.setText(folder)
        self._load_folder(Path(folder))

    def _load_folder(self, folder: Path) -> None:
        exts = {".jpg", ".jpeg", ".png"}
        self.image_paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)
        self.folder_path = folder
        self.image_combo.clear()
        for p in self.image_paths:
            self.image_combo.addItem(p.name)
        if self.image_paths:
            self.current_index = 0
            self.image_combo.setCurrentIndex(0)
            self._load_image(self.image_paths[0])
        else:
            self.current_index = -1
            self.canvas.set_image(None)
            self.status_label.setText("No supported images found in selected folder.")

    def _on_image_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.image_paths):
            return
        self.current_index = idx
        self._load_image(self.image_paths[idx])

    def _load_image(self, path: Path) -> None:
        try:
            with Image.open(path) as src:
                self.src_exif_bytes = src.info.get("exif")
                self.src_icc_profile = src.info.get("icc_profile")
                self.src_quantization = getattr(src, "quantization", None)
                img = ImageOps.exif_transpose(src).convert("RGB")
            arr = np.array(img)
            self.current_image_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.current_ext = path.suffix.lower()
            self.canvas.set_image(self.current_image_bgr)
            self.status_label.setText(f"Loaded {path.name}. Draw boxes for missed objects.")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _prev_image(self) -> None:
        if not self.image_paths:
            return
        i = max(0, self.current_index - 1)
        self.image_combo.setCurrentIndex(i)

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        i = min(len(self.image_paths) - 1, self.current_index + 1)
        self.image_combo.setCurrentIndex(i)

    def _apply_effect(self) -> None:
        if self.current_image_bgr is None:
            return

        rects = self.canvas.get_rects()
        if not rects:
            QMessageBox.information(self, "No boxes", "Draw at least one box first.")
            return

        arr = self.current_image_bgr.copy()
        effect = self.effect_combo.currentText()

        if effect == "blur":
            sigma = self.blur_sigma_spin.value()
            ksize = sigma * 2 + 1
            for x, y, w, h in rects:
                roi = arr[y:y + h, x:x + w]
                if roi.size > 0:
                    arr[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        elif effect == "pixelate":
            block = self.pixelate_block_spin.value()
            for x, y, w, h in rects:
                roi = arr[y:y + h, x:x + w]
                if roi.size == 0:
                    continue
                small_w = max(1, w // block)
                small_h = max(1, h // block)
                small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                arr[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            radius = self.inpaint_radius_spin.value()
            h, w = arr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for x, y, rw, rh in rects:
                cv2.rectangle(mask, (x, y), (x + rw, y + rh), 255, thickness=-1)
            arr = cv2.inpaint(arr, mask, radius, cv2.INPAINT_TELEA)

        self.current_image_bgr = arr
        self.canvas.set_image(self.current_image_bgr)
        self.status_label.setText(f"Applied {effect} to {len(rects)} box(es).")

    def _save_current(self) -> None:
        if self.current_image_bgr is None or self.current_index < 0:
            return

        path = self.image_paths[self.current_index]
        try:
            rgb = cv2.cvtColor(self.current_image_bgr, cv2.COLOR_BGR2RGB)
            out_img = Image.fromarray(rgb)

            out_exif = build_output_exif(self.src_exif_bytes, strip_gps=self.strip_gps_cb.isChecked())
            save_kwargs: dict = {}
            if out_exif:
                save_kwargs["exif"] = out_exif
            if self.src_icc_profile:
                save_kwargs["icc_profile"] = self.src_icc_profile

            fmt = "JPEG" if self.current_ext in {".jpg", ".jpeg"} else "PNG"
            if fmt == "JPEG":
                if self.src_quantization:
                    save_kwargs["qtables"] = self.src_quantization
                else:
                    save_kwargs["quality"] = 92
                save_kwargs.setdefault("optimize", True)

            tmp_path = path.with_name(path.name + ".image_cleaner.tmp")
            try:
                out_img.save(tmp_path, format=fmt, **save_kwargs)
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

            self.status_label.setText(f"Saved {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))


class ImageCleanerGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Cleaner - GUI")
        self.resize(1100, 780)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_process_output)
        self.process.finished.connect(self._on_process_finished)

        self.script_path_input = QLineEdit(str(DEFAULT_SCRIPT_PATH))
        self.subpath_input = QLineEdit()
        self.subpath_input.setPlaceholderText(r"Example: Camera\Design\Camera Views")

        self.btn_browse_script = QPushButton("Browse")
        self.btn_browse_script.clicked.connect(self._browse_script)

        self.effect_combo = QComboBox()
        self.effect_combo.addItems(["auto", "inpaint", "blur", "pixelate"])

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 128)
        self.workers_spin.setValue(max(1, (os.cpu_count() or 2) - 1))

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.40)

        self.blur_sigma_spin = QSpinBox()
        self.blur_sigma_spin.setRange(1, 500)
        self.blur_sigma_spin.setValue(45)

        self.inpaint_radius_spin = QSpinBox()
        self.inpaint_radius_spin.setRange(1, 500)
        self.inpaint_radius_spin.setValue(15)

        self.pixelate_block_spin = QSpinBox()
        self.pixelate_block_spin.setRange(1, 500)
        self.pixelate_block_spin.setValue(20)

        self.arrow_filter_input = QLineEdit()
        self.arrow_filter_input.setPlaceholderText("Optional, e.g. _INSTALL")

        self.log_file_input = QLineEdit()
        self.log_file_input.setPlaceholderText("Optional log file path")

        self.btn_browse_log = QPushButton("Browse")
        self.btn_browse_log.clicked.connect(self._browse_log_file)

        self.recursive_cb = QCheckBox("Recursive")
        self.backup_cb = QCheckBox("Backup originals")
        self.strip_gps_cb = QCheckBox("Strip GPS")
        self.force_cb = QCheckBox("Force reprocess")
        self.dry_run_cb = QCheckBox("Dry run")
        self.verbose_cb = QCheckBox("Verbose")
        self.manual_touchup_cb = QCheckBox("After run, open manual touch-up for missed objects")

        self.btn_run = QPushButton("Run")
        self.btn_run.clicked.connect(self._run_cleaner)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self._stop_cleaner)
        self.btn_stop.setEnabled(False)

        self.btn_clear = QPushButton("Clear Log")
        self.btn_clear.clicked.connect(self._clear_log)

        self.btn_manual_touchup = QPushButton("Manual Touch-up")
        self.btn_manual_touchup.clicked.connect(self._open_manual_touchup)

        self.output_view = QPlainTextEdit()
        self.output_view.setReadOnly(True)

        self.status_label = QLabel("Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        script_row = QHBoxLayout()
        script_row.addWidget(self.script_path_input)
        script_row.addWidget(self.btn_browse_script)

        log_row = QHBoxLayout()
        log_row.addWidget(self.log_file_input)
        log_row.addWidget(self.btn_browse_log)

        basic_group = QGroupBox("Required")
        basic_form = QFormLayout()
        basic_form.addRow("Cleaner script", self._wrap(script_row))
        basic_form.addRow("Project subpath", self.subpath_input)
        basic_group.setLayout(basic_form)

        options_group = QGroupBox("Options")
        options_form = QFormLayout()
        options_form.addRow("Effect", self.effect_combo)
        options_form.addRow("Workers", self.workers_spin)
        options_form.addRow("Confidence", self.conf_spin)
        options_form.addRow("Blur sigma", self.blur_sigma_spin)
        options_form.addRow("Inpaint radius", self.inpaint_radius_spin)
        options_form.addRow("Pixelate block", self.pixelate_block_spin)
        options_form.addRow("Arrow filter", self.arrow_filter_input)
        options_form.addRow("Log file", self._wrap(log_row))
        options_group.setLayout(options_form)

        flags_group = QGroupBox("Flags")
        flags_layout = QGridLayout()
        flags_layout.addWidget(self.recursive_cb, 0, 0)
        flags_layout.addWidget(self.backup_cb, 0, 1)
        flags_layout.addWidget(self.strip_gps_cb, 1, 0)
        flags_layout.addWidget(self.force_cb, 1, 1)
        flags_layout.addWidget(self.dry_run_cb, 2, 0)
        flags_layout.addWidget(self.verbose_cb, 2, 1)
        flags_layout.addWidget(self.manual_touchup_cb, 3, 0, 1, 2)
        flags_group.setLayout(flags_layout)

        button_row = QHBoxLayout()
        button_row.addWidget(self.btn_run)
        button_row.addWidget(self.btn_stop)
        button_row.addWidget(self.btn_manual_touchup)
        button_row.addWidget(self.btn_clear)

        main_layout = QVBoxLayout()
        main_layout.addWidget(basic_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(flags_group)
        main_layout.addLayout(button_row)
        main_layout.addWidget(QLabel("Output"))
        main_layout.addWidget(self.output_view, 1)
        main_layout.addWidget(self.status_label)

        root = QWidget()
        root.setLayout(main_layout)
        self.setCentralWidget(root)

    def _wrap(self, layout: QHBoxLayout) -> QWidget:
        container = QWidget()
        container.setLayout(layout)
        return container

    def _browse_script(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image_cleaner.py",
            str(Path.cwd()),
            "Python Files (*.py)",
        )
        if file_path:
            self.script_path_input.setText(file_path)

    def _browse_log_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select log file",
            str(Path.cwd() / "image_cleaner.log"),
            "Log Files (*.log);;All Files (*)",
        )
        if file_path:
            self.log_file_input.setText(file_path)

    def _run_cleaner(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Running", "A cleaning job is already running.")
            return

        script_path = Path(self.script_path_input.text().strip())
        if not script_path.exists() or not script_path.is_file():
            QMessageBox.warning(self, "Invalid script", "Select a valid image_cleaner.py file.")
            return

        subpath = self.subpath_input.text().strip()
        if not subpath:
            QMessageBox.warning(self, "Missing subpath", "Enter a project subpath.")
            return

        if Path(subpath).is_absolute():
            QMessageBox.warning(self, "Invalid subpath", "Subpath must be relative, not absolute.")
            return

        args: list[str] = [
            str(script_path),
            subpath,
            "--effect",
            self.effect_combo.currentText(),
            "--workers",
            str(self.workers_spin.value()),
            "--conf-thresh",
            f"{self.conf_spin.value():.2f}",
            "--blur-sigma",
            str(self.blur_sigma_spin.value()),
            "--inpaint-radius",
            str(self.inpaint_radius_spin.value()),
            "--pixelate-block",
            str(self.pixelate_block_spin.value()),
        ]

        arrow_filter = self.arrow_filter_input.text().strip()
        if arrow_filter:
            args.extend(["--arrow-filter", arrow_filter])

        log_file = self.log_file_input.text().strip()
        if log_file:
            args.extend(["--log-file", log_file])

        if self.recursive_cb.isChecked():
            args.append("--recursive")
        if self.backup_cb.isChecked():
            args.append("--backup")
        if self.strip_gps_cb.isChecked():
            args.append("--strip-gps")
        if self.force_cb.isChecked():
            args.append("--force")
        if self.dry_run_cb.isChecked():
            args.append("--dry-run")
        if self.verbose_cb.isChecked():
            args.append("--verbose")

        self.output_view.appendPlainText("Running command:")
        self.output_view.appendPlainText(f"{sys.executable} {' '.join(args)}")
        self.output_view.appendPlainText("-" * 80)

        self.process.setWorkingDirectory(str(script_path.parent))
        self.process.start(sys.executable, args)

        if not self.process.waitForStarted(3000):
            QMessageBox.critical(self, "Failed to start", "Could not start image_cleaner.py")
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Running")

    def _stop_cleaner(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return

        self.output_view.appendPlainText("Stopping process...")
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    def _on_process_output(self) -> None:
        data = self.process.readAllStandardOutput().data().decode(errors="replace")
        if data:
            self.output_view.appendPlainText(data.rstrip("\n"))

    def _on_process_finished(self, exit_code: int, _exit_status) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if exit_code == 0:
            self.status_label.setText("Completed successfully")
        else:
            self.status_label.setText(f"Finished with exit code {exit_code}")
        self.output_view.appendPlainText("-" * 80)
        self.output_view.appendPlainText(f"Process finished with exit code {exit_code}")

        if exit_code == 0 and self.manual_touchup_cb.isChecked():
            self._open_manual_touchup()

    def _open_manual_touchup(self) -> None:
        dlg = ManualTouchupDialog(self, strip_gps_default=self.strip_gps_cb.isChecked())
        dlg.exec()

    def _clear_log(self) -> None:
        self.output_view.clear()


def main() -> None:
    app = QApplication(sys.argv)
    win = ImageCleanerGui()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
