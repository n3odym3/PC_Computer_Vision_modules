from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg
import numpy as np

from core.window_base import WindowBase
from core.input_output_types import IOTypes
from modules.computer_vision.vignetting_corrector.vignetting_corrector import VignettingCorrector


class Vignetting_corrector_win(WindowBase):
    """
    Vignetting Corrector window module.

    Loads a flat-field reference calibration frame and corrects radial illumination
    falloff (vignetting) across 8-bit, 12-bit, or 16-bit monochrome and color frames.
    """

    _persistent_fields: List[str]
    accepted_input_types: List[IOTypes]
    outputs: Dict[str, IOTypes]
    connections: Dict[str, List[Any]]
    processor: VignettingCorrector
    _last_raw_frame: Optional[Any]
    _last_corrected_frame: Optional[Any]
    _is_comparing: bool
    info_text_tag: str
    enable_checkbox_tag: str
    depth_combo_tag: str
    compare_btn_tag: str
    item_handler_tag: str

    def __init__(
        self,
        label: str = "Vignetting Corrector",
        win_width: int = 250,
        win_height: int = 180,
        pos: Tuple[int, int] = (20, 20),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Vignetting Corrector window.

        Args:
            label: Window display title.
            win_width: Width in pixels.
            win_height: Height in pixels.
            pos: (x, y) coordinates.
            uuid: Unique identifier.
            outputs: Optional custom outputs mapping.
            visible: Initial visibility state.
            **kwargs: Extra arguments.
        """
        super().__init__(
            label=label,
            pos=pos,
            win_width=win_width,
            win_height=win_height,
            uuid=uuid,
            outputs=outputs,
            visible=visible,
            **kwargs,
        )

        self._persistent_fields = ["label"]
        self.accepted_input_types = [IOTypes.FRAME, IOTypes.FILE_PATH]

        self.outputs = outputs or {
            "Frame": IOTypes.FRAME,
        }
        self.connections = {k: [] for k in self.outputs}

        self.processor = VignettingCorrector()

        self._last_raw_frame = None
        self._last_corrected_frame = None
        self._is_comparing = False

        self.info_text_tag = f"vignetting_info_{self.UUID}"
        self.enable_checkbox_tag = f"vignetting_enable_{self.UUID}"
        self.depth_combo_tag = f"vignetting_depth_{self.UUID}"
        self.compare_btn_tag = f"vignetting_compare_{self.UUID}"
        self.item_handler_tag = f"vignetting_comp_hand_{self.UUID}"
        self.handler_tag = f"vignetting_handlers_{self.UUID}"

        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            dpg.add_text("No reference loaded", tag=self.info_text_tag, wrap=self.win_width - 20)
            dpg.add_button(label="Select Reference File", callback=self.browse_file, width=-1)

            dpg.add_separator()
            dpg.add_checkbox(
                label="Enable Correction",
                default_value=True,
                tag=self.enable_checkbox_tag,
                callback=self._on_toggle_enable,
            )

            with dpg.group(horizontal=True):
                dpg.add_text("Matrix bit depth:")
                dpg.add_combo(
                    ["8-bit", "12-bit", "16-bit"],
                    default_value="8-bit",
                    tag=self.depth_combo_tag,
                    width=80,
                    callback=self._on_toggle_enable,
                )

            dpg.add_button(label="Send matrix as Preview", callback=self.send_matrix_preview, width=-1)

            dpg.add_separator()
            dpg.add_button(label="COMPARE (Hold)", tag=self.compare_btn_tag, width=-1, height=30)

        with dpg.item_handler_registry(tag=self.item_handler_tag):
            dpg.add_item_clicked_handler(callback=self._compare_on)
            dpg.add_item_deactivated_handler(callback=self._compare_off)
        dpg.bind_item_handler_registry(self.compare_btn_tag, self.item_handler_tag)

        with dpg.handler_registry(tag=self.handler_tag):
            dpg.add_mouse_release_handler(callback=self._compare_off)

        self.update_permission()

    def update_permission(self) -> None:
        """Update UI permissions based on application mode."""
        from core.app_state import app_state
        is_user = (app_state.mode == "user")

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _on_toggle_enable(self, sender: Any = None, app_data: Any = None, *args: Any, **kwargs: Any) -> None:
        """Immediately re-evaluate and emit last frame when checkbox or depth is changed."""
        if self._last_raw_frame is not None and not self._is_comparing:
            enabled = dpg.get_value(self.enable_checkbox_tag)
            depth = dpg.get_value(self.depth_combo_tag)
            self._last_corrected_frame = self.processor.correct_frame(
                self._last_raw_frame, enabled=enabled, depth_str=depth
            )
            frame_to_send = self._last_corrected_frame if enabled else self._last_raw_frame
            self.trigger_cb(frame=frame_to_send)

    def _compare_on(self, *args: Any, **kwargs: Any) -> None:
        """Switch output stream to uncorrected raw frame while button is held."""
        self._is_comparing = True
        if self._last_raw_frame is not None:
            self.trigger_cb(self._last_raw_frame)

    def _compare_off(self, *args: Any, **kwargs: Any) -> None:
        """Revert output stream to corrected frame when comparison button released."""
        if not self._is_comparing:
            return
        self._is_comparing = False
        enabled = dpg.get_value(self.enable_checkbox_tag)
        depth = dpg.get_value(self.depth_combo_tag)
        if self._last_raw_frame is not None:
            self._last_corrected_frame = self.processor.correct_frame(
                self._last_raw_frame, enabled=enabled, depth_str=depth
            )
            frame_to_send = self._last_corrected_frame if enabled else self._last_raw_frame
            self.trigger_cb(frame=frame_to_send)
        elif self._last_corrected_frame is not None:
            self.trigger_cb(self._last_corrected_frame)

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming FRAME or FILE_PATH data.

        Args:
            *args: Positional frame input.
            **kwargs: Keyword arguments containing data or path.
        """
        data = kwargs.get("data")
        if data is None:
            data = kwargs.get("frame")
        if data is None:
            data = kwargs.get("path")
        if data is None:
            data = args[0] if args else None
        data_type = kwargs.get("data_type")

        if data_type == IOTypes.FILE_PATH:
            self.load_reference(data)
            return

        if isinstance(data, str) and os.path.exists(data):
            self.load_reference(data)
            return

        if data is not None:
            enabled = dpg.get_value(self.enable_checkbox_tag)
            depth = dpg.get_value(self.depth_combo_tag)

            self._last_raw_frame = data
            frame_corrected = self.processor.correct_frame(data, enabled=enabled, depth_str=depth)
            self._last_corrected_frame = frame_corrected

            if self._is_comparing:
                self.trigger_cb(frame=self._last_raw_frame)
            else:
                self.trigger_cb(frame=self._last_corrected_frame)

    def load_reference(self, path: str) -> None:
        """
        Load flat-field reference calibration image from filesystem.

        Args:
            path: Absolute or relative file path to reference image.
        """
        if self.processor.load_reference(path):
            dpg.set_value(self.info_text_tag, f"Ref: {os.path.basename(path)}")
        else:
            dpg.set_value(self.info_text_tag, "Failed to load reference")

    def browse_file(self, *args: Any, **kwargs: Any) -> None:
        """Open system file dialog to select a vignetting reference image."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.call("wm", "attributes", ".", "-topmost", True)
            file_path = filedialog.askopenfilename(
                title="Select Reference File",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
            )
            if file_path:
                self.load_reference(file_path)
        except Exception as e:
            from loguru import logger
            logger.error(f"Error opening file dialog: {e}")

    def send_matrix_preview(self, *args: Any, **kwargs: Any) -> None:
        """Sends the correction matrix scaled to the required bit-depth to the outputs for visual inspection."""
        depth = dpg.get_value(self.depth_combo_tag)
        preview_frame = self.processor.get_correction_preview(depth_str=depth)

        if preview_frame is not None:
            self.trigger_cb(frame=(preview_frame, f"Vignetting_Matrix_{depth}"))

    def is_ready(self) -> bool:
        """Indicate readiness for input frames."""
        return True

    def is_outputs_ready(self) -> bool:
        """Indicate output port readiness."""
        return True

    def trigger_cb(self, frame: Any) -> None:
        """
        Dispatch frame to connected downstream modules.

        Args:
            frame: Processed image array or (array, name) tuple.
        """
        for idx, output_key in enumerate(self.outputs):
            connected_modules = self.connections.get(output_key, [])
            for module in connected_modules:
                if idx == 0:
                    module.input_cb(data=frame, data_type=IOTypes.FRAME)

    def close(self) -> None:
        """Clean up registered DearPyGui handlers upon closing."""
        if dpg.does_item_exist(self.item_handler_tag):
            dpg.delete_item(self.item_handler_tag)
        if dpg.does_item_exist(self.handler_tag):
            dpg.delete_item(self.handler_tag)
        super().close()

    def destroy(self) -> None:
        """Clean up registered DearPyGui handlers upon deletion."""
        self.close()
        super().destroy()


VignettingCorrectorWin = Vignetting_corrector_win
EXPORTED_CLASS = Vignetting_corrector_win
EXPORTED_NAME = "Vignetting Corrector"
