from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import dearpygui.dearpygui as dpg
import cv2
import numpy as np
from loguru import logger

from core.window_base import WindowBase
from core.input_output_types import IOTypes

try:
    from pyzbar.pyzbar import decode
    ZBAR_AVAILABLE = True
except ImportError:
    ZBAR_AVAILABLE = False
    logger.warning("pyzbar not installed. Barcode reading will be limited to OpenCV QR Code detection. Run 'pip install pyzbar' for better support.")


class BarcodeReaderWin(WindowBase):
    """
    Barcode and QR Code Reader window.

    Processes incoming video/image frames, decodes 1D barcodes and 2D QR codes
    using pyzbar (or OpenCV QRCodeDetector as fallback), displays decoded content,
    and emits the decoded string to downstream modules.
    """

    _persistent_fields: List[str]
    accepted_input_types: List[IOTypes]
    outputs: Dict[str, IOTypes]
    connections: Dict[str, List[Any]]
    last_decoded: str
    qr_detector: Optional[cv2.QRCodeDetector]
    text_tag: str
    status_tag: str

    def __init__(
        self,
        label: str = "Barcode / QR Reader",
        win_width: int = 300,
        win_height: int = 180,
        pos: Tuple[int, int] = (100, 100),
        uuid: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Barcode / QR Reader window.

        Args:
            label: Window display title.
            win_width: Initial window width in pixels.
            win_height: Initial window height in pixels.
            pos: (x, y) coordinates for initial placement.
            uuid: Unique identifier for node persistence.
            outputs: Optional custom outputs mapping.
            visible: Whether the window starts visible.
            **kwargs: Extra persistent configuration fields.
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
        self.accepted_input_types = [IOTypes.FRAME]
        self.outputs = outputs or {
            "Decoded Text": IOTypes.TEXT,
        }
        self.connections = {k: [] for k in self.outputs}

        self.last_decoded = ""
        self.qr_detector = cv2.QRCodeDetector() if not ZBAR_AVAILABLE else None

        self.text_tag = f"barcode_text_{self.UUID}"
        self.status_tag = f"barcode_status_{self.UUID}"

        with dpg.window(
            label=self.label,
            width=self.win_width,
            height=self.win_height,
            pos=self.pos,
            tag=self.winID,
            show=self.visible,
        ):
            if not ZBAR_AVAILABLE:
                dpg.add_text("Warning: pyzbar not found.", color=(255, 100, 100))
                dpg.add_text("Limited to OpenCV QR Code reading.")

            dpg.add_text("Last Decoded:", color=(200, 200, 200))
            dpg.add_input_text(tag=self.text_tag, readonly=True, width=-1, multiline=True, height=50)
            dpg.add_text("Status: Waiting for frames...", tag=self.status_tag, color=(100, 255, 100))

        self.update_permission()

    def update_permission(self) -> None:
        """Update UI permissions based on application mode."""
        from core.app_state import app_state
        is_user = (app_state.mode == "user")

        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Indicate whether the module is ready to receive frames."""
        return True

    def input_cb(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming frame data from upstream modules.

        Args:
            *args: Positional frame input.
            **kwargs: Keyword arguments containing 'data'.
        """
        frame = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)

        if isinstance(frame, (tuple, list)) and len(frame) >= 1:
            frame = frame[0]

        if not isinstance(frame, np.ndarray):
            return

        self.process_frame(frame)

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Decode barcode or QR code from frame and emit string if found.

        Args:
            frame: Numpy image array (grayscale, BGR, or 16-bit).
        """
        try:
            decoded_text: Optional[str] = None

            gray = frame
            if gray.dtype != np.uint8:
                if gray.dtype == np.uint16:
                    gray = (gray >> 8).astype(np.uint8)
                else:
                    gray = gray.astype(np.uint8)

            if frame.ndim == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

            if ZBAR_AVAILABLE:
                decoded_objects = decode(gray)
                if decoded_objects:
                    decoded_text = decoded_objects[0].data.decode("utf-8")
            else:
                if self.qr_detector:
                    data, bbox, _ = self.qr_detector.detectAndDecode(gray)
                    if data:
                        decoded_text = data
                    elif hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
                        bd = cv2.barcode.BarcodeDetector()
                        barcode_res = bd.detectAndDecode(gray)
                        if barcode_res and len(barcode_res) >= 1 and barcode_res[0]:
                            info = barcode_res[0]
                            if isinstance(info, (list, tuple)) and len(info) > 0 and info[0]:
                                decoded_text = str(info[0])
                            elif isinstance(info, str) and info:
                                decoded_text = info

            if decoded_text:
                if decoded_text != self.last_decoded:
                    self.last_decoded = decoded_text
                    dpg.set_value(self.text_tag, self.last_decoded)
                    dpg.configure_item(self.status_tag, default_value="Status: Code Detected!", color=(100, 255, 100))
                    self.emit_result(self.last_decoded)
            else:
                dpg.configure_item(self.status_tag, default_value="Status: No code found.", color=(255, 150, 100))

        except Exception as e:
            logger.error(f"Barcode decoding error: {e}")

    def emit_result(self, text: str) -> None:
        """
        Send decoded text payload to all downstream connected modules.

        Args:
            text: Decoded string payload.
        """
        for module in self.connections.get("Decoded Text", []):
            try:
                module.input_cb(data=text, data_type=IOTypes.TEXT)
            except Exception as e:
                logger.error(f"Error sending to {module.label}: {e}")


EXPORTED_CLASS = BarcodeReaderWin
EXPORTED_NAME = "Barcode / QR Reader"
