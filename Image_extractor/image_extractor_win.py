from __future__ import annotations

import io
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dearpygui.dearpygui as dpg
from loguru import logger
import numpy as np

from core.input_output_types import IOTypes
from core.window_base import WindowBase

try:
    from PIL import Image
    import win32clipboard as clip
    CLIPBOARD_ENABLED = True
except ImportError:
    logger.warning("Pillow or pywin32 not found. Clipboard export will be disabled.")
    CLIPBOARD_ENABLED = False


class ImageExtractorWin(WindowBase):
    """
    Image Extractor Module.

    Takes a (Frame, Mask) pair, extracts the masked region, sets the background
    to transparent (RGBA), displays the result in an interactive viewer, and
    exports it to the Windows clipboard (PNG / DIB) automatically or on demand.
    """

    _persistent_fields: List[str]
    accepted_input_types: List[IOTypes]
    outputs: Dict[str, IOTypes]
    connections: Dict[str, List[Any]]
    _last_rgba: Optional[np.ndarray]
    _auto_copy: bool
    _tag_texture: str
    _tag_texture_reg: str
    _tag_plot: str
    _tag_status: str
    texture_width: int
    texture_height: int

    def __init__(
        self,
        label: str = "Image Extractor",
        win_width: int = 400,
        win_height: int = 500,
        pos: Tuple[int, int] = (500, 200),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Image Extractor module window.

        Args:
            label: Window display label.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: (x, y) initial placement.
            uuid: Unique identifier.
            outputs: Optional custom outputs mapping.
            visible: Whether the window starts visible.
            **kwargs: Extra arguments.
        """
        outputs = outputs or {
            "Extracted RGBA": IOTypes.FRAME,
        }

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
        self.accepted_input_types = [IOTypes.FRAME_MASK_PAIR]
        self.connections = {k: [] for k in self.outputs}

        self._last_rgba = None
        self._auto_copy = True

        self._tag_texture = f"extractor_texture_{self.UUID}"
        self._tag_texture_reg = f"extractor_texture_reg_{self.UUID}"
        self._tag_plot = f"extractor_plot_{self.UUID}"
        self._tag_status = f"extractor_status_{self.UUID}"

        self.texture_width = 100
        self.texture_height = 100

        dpg.add_texture_registry(tag=self._tag_texture_reg, show=False)
        self._build_ui()
        self.update_permission()

    def update_permission(self) -> None:
        """Update UI permissions based on application mode."""
        from core.app_state import app_state
        is_user = (app_state.mode == "user")

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Indicate whether the extractor is ready for inputs."""
        return True

    def _build_ui(self) -> None:
        """Construct the DearPyGui window layout and preview plot."""
        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Auto-Copy to Clipboard",
                    default_value=self._auto_copy,
                    callback=lambda s, v: setattr(self, "_auto_copy", v),
                )
                dpg.add_button(label="Manual Copy", callback=self._manual_copy_cb)

            dpg.add_text("Ready", tag=self._tag_status, color=(100, 255, 100))
            dpg.add_separator()

            self._init_viewer(300, 200)

    def _init_viewer(self, width: int, height: int) -> None:
        """Initialize or resize the preview texture and DPG plot."""
        if dpg.does_item_exist(self._tag_texture):
            dpg.delete_item(self._tag_texture)
        if dpg.does_item_exist(self._tag_plot):
            dpg.delete_item(self._tag_plot)

        self.texture_width = width
        self.texture_height = height

        img = np.zeros((height, width, 4), dtype=np.uint8)

        dpg.add_raw_texture(
            width,
            height,
            self._convert_to_dpg(img),
            tag=self._tag_texture,
            format=dpg.mvFormat_Float_rgba,
            parent=self._tag_texture_reg,
        )

        plot_tag = dpg.add_plot(
            tag=self._tag_plot,
            no_menus=True,
            no_title=True,
            width=-1,
            height=-1,
            parent=self.winID,
        )
        dpg.add_plot_axis(
            dpg.mvXAxis,
            no_gridlines=True,
            no_tick_labels=True,
            no_tick_marks=True,
            parent=plot_tag,
        )
        yaxis = dpg.add_plot_axis(
            dpg.mvYAxis,
            no_gridlines=True,
            no_tick_labels=True,
            no_tick_marks=True,
            parent=plot_tag,
        )
        dpg.add_image_series(self._tag_texture, [0, 0], [width, height], parent=yaxis)

    def _convert_to_dpg(self, img: np.ndarray) -> np.ndarray:
        """Convert RGBA image to flattened float array for DearPyGui raw textures."""
        return (img.ravel().astype(np.float32)) / 255.0

    def input_cb(self, data: Any, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming FRAME_MASK_PAIR data.

        Args:
            data: Tuple of (frame, mask).
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        if not (isinstance(data, tuple) and len(data) == 2):
            return

        frame, mask = data
        if not (isinstance(frame, np.ndarray) and isinstance(mask, np.ndarray)):
            return

        try:
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif len(frame.shape) == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb = frame
                if rgb.shape[2] == 4:
                    rgb = rgb[:, :, :3]

            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = rgb
            rgba[:, :, 3] = mask_binary

            self._last_rgba = rgba

            if rgba.shape[1] != self.texture_width or rgba.shape[0] != self.texture_height:
                self._init_viewer(rgba.shape[1], rgba.shape[0])

            dpg.set_value(self._tag_texture, self._convert_to_dpg(rgba))

            if self._auto_copy:
                threading.Thread(target=self._copy_to_clip, args=(rgba,), daemon=True).start()

            for module in self.connections.get("Extracted RGBA", []):
                module.input_cb(data=rgba, data_type=IOTypes.FRAME)

            dpg.set_value(self._tag_status, "Exported" if self._auto_copy else "Ready")
            dpg.configure_item(self._tag_status, color=(100, 255, 100))

        except Exception as e:
            logger.error(f"Extraction Error: {e}")
            dpg.set_value(self._tag_status, f"Error: {e}")
            dpg.configure_item(self._tag_status, color=(255, 100, 100))

    def _manual_copy_cb(self, *args: Any, **kwargs: Any) -> None:
        """Callback to manually copy current RGBA image to clipboard."""
        if self._last_rgba is not None:
            threading.Thread(target=self._copy_to_clip, args=(self._last_rgba,), daemon=True).start()
            dpg.set_value(self._tag_status, "Copied manually")
        else:
            dpg.set_value(self._tag_status, "No image to copy")

    def _copy_to_clip(self, rgba_image: np.ndarray) -> None:
        """Copy RGBA image buffer to Windows clipboard as PNG and CF_DIB."""
        if not CLIPBOARD_ENABLED:
            logger.error("Clipboard export disabled (PIL or pywin32 missing)")
            return

        try:
            pil_img = Image.fromarray(rgba_image, "RGBA")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            data = buffer.getvalue()
            buffer.close()

            clip.OpenClipboard()
            try:
                clip.EmptyClipboard()
                png_format = clip.RegisterClipboardFormat("PNG")
                if png_format:
                    clip.SetClipboardData(png_format, data)

                bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[3])
                dib_buffer = io.BytesIO()
                bg.save(dib_buffer, format="BMP")
                dib_data = dib_buffer.getvalue()[14:]
                dib_buffer.close()
                clip.SetClipboardData(clip.CF_DIB, dib_data)
            finally:
                clip.CloseClipboard()
            logger.debug("Image copied to clipboard as PNG.")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")

    def destroy(self) -> None:
        """Clean up texture registry on window destruction."""
        if dpg.does_item_exist(self._tag_texture_reg):
            dpg.delete_item(self._tag_texture_reg)
        super().destroy()


EXPORTED_CLASS = ImageExtractorWin
EXPORTED_NAME = "Image Extractor"
