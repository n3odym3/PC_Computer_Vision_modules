import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from loguru import logger
import numpy as np
import cv2
import threading
from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- SAM CONFIGURATION ---
# Modify these variables to point to your weight file
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth" #"vit_b_coralscop.pth"
SAM_MODEL_TYPE = "vit_b"
# -------------------------

class SamInteractiveWin(WindowBase):
    def __init__(self,
                 label="SAM Interactive",
                 win_width=600,
                 win_height=650,
                 pos=(50, 50),
                 uuid=None,
                 outputs=None,
                 visible=True):

        outputs = outputs or {
            "Overlay": IOTypes.FRAME,
            "Mask": IOTypes.FRAME,
            "ROI": IOTypes.ROI_SAMPLE,
            "Result": IOTypes.FRAME_MASK_PAIR,
        }

        super().__init__(label=label, pos=pos,
                         win_width=win_width, win_height=win_height,
                         uuid=uuid, outputs=outputs, visible=visible)

        self.accepted_input_types = [IOTypes.FRAME, IOTypes.FRAME12, IOTypes.FRAME16, IOTypes.CMD_DICT]
        self._persistent_fields = ["label"]
        self.enabled = True

        self._frame: np.ndarray | None = None
        self._rgb_frame: np.ndarray | None = None
        self._raw_bgr_frame: np.ndarray | None = None
        self._process_enabled = False
        self._saturation = 1.0
        self._gamma = 1.0
        self._current_mask = None

        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()

        # UI Tags for image processing controls
        self._tag_proc_enable = f"sam_proc_enable_{self.UUID}"
        self._tag_proc_sat = f"sam_proc_sat_{self.UUID}"
        self._tag_proc_gamma = f"sam_proc_gamma_{self.UUID}"

        # Model initialization (loaded once at instantiation)
        self.predictor = None
        self._load_sam_model()
        
        self.points = []
        self.labels = []
        
        # UI Tags
        self._tag_plot = f"sam_plot_{self.UUID}"
        self._tag_texture = f"sam_texture_{self.UUID}"
        self._tag_texture_reg = f"sam_texture_reg_{self.UUID}"
        dpg.add_texture_registry(tag=self._tag_texture_reg, show=False)

        self._tag_status = f"sam_status_{self.UUID}"
        self._tag_reset_btn = f"sam_reset_{self.UUID}"
        
        self.texture_width = 100
        self.texture_height = 100
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_hovering = False
        self._tag_handler_reg = f"sam_handlers_{self.UUID}"

        self._build_ui()

        self.update_permission()

    def update_permission(self):
        """
        Adjust module permissions and UI visibility based on app_state.mode.
        Disable closing in user mode.
        """
        from core.app_state import app_state
        mode = app_state.mode
        
        if dpg.does_item_exist(self.winID):
            is_user = (mode == "user")
            dpg.configure_item(self.winID, no_close=is_user)

    def is_ready(self) -> bool:
        """Return True if model predictor is loaded and ready."""
        return self.predictor is not None

    def close(self) -> None:
        """Clean up registered DearPyGui handlers and textures upon closing."""
        if dpg.does_item_exist(self._tag_handler_reg):
            dpg.delete_item(self._tag_handler_reg)
        if dpg.does_item_exist(self._tag_texture_reg):
            dpg.delete_item(self._tag_texture_reg)
        super().close()

    def destroy(self) -> None:
        """Destroy module resources."""
        self.close()
        super().destroy()

    def _load_sam_model(self):
        """Loads the SAM model when the module is created"""
        try:
            ckpt_path = Path(__file__).parent / SAM_CHECKPOINT_NAME
            if not ckpt_path.exists():
                logger.error(f"Weight file not found: {ckpt_path}")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing SAM ({SAM_MODEL_TYPE}) on {device}...")
            
            # To support CAT (Category-Aware) models which have extra keys (cate_...),
            # we instantiate the model and then load the weights with strict=False.
            sam = sam_model_registry[SAM_MODEL_TYPE]()
            state_dict = torch.load(str(ckpt_path), map_location=device)
            
            # Flexible loading to ignore unused CAT classification layers here
            msg = sam.load_state_dict(state_dict, strict=False)
            if msg.unexpected_keys:
                logger.warning(f"SAM: Some layers ({len(msg.unexpected_keys)}) were ignored (specific CAT layers).")
            
            sam.to(device=device)
            self.predictor = SamPredictor(sam)
            logger.success("SAM loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SAM: {e}")

    def _build_ui(self):
        with dpg.window(label=self.label, width=self.win_width, height=self.win_height,
                        pos=self.pos, tag=self.winID, show=self.visible):
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Reset Points", tag=self._tag_reset_btn, callback=self.reset_points)
                dpg.add_text("Ready", tag=self._tag_status, color=(100, 200, 100))

            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Enable Proc.", tag=self._tag_proc_enable, default_value=self._process_enabled, callback=self._on_proc_settings_change)
                dpg.add_text("Sat:")
                dpg.add_drag_float(tag=self._tag_proc_sat, default_value=self._saturation, min_value=0.0, max_value=5.0, speed=0.05, width=80, callback=self._on_proc_settings_change)
                dpg.add_text("Gamma:")
                dpg.add_drag_float(tag=self._tag_proc_gamma, default_value=self._gamma, min_value=0.1, max_value=5.0, speed=0.05, width=80, callback=self._on_proc_settings_change)

            with dpg.group(horizontal=True):
                dpg.add_text("CTRL+Left Click: Include", color=(150, 255, 150))
                dpg.add_text(" CTRL+Right Click: Exclude", color=(255, 150, 150))
            dpg.add_separator()
            
            self._init_viewer(800, 600)

            with dpg.handler_registry(tag=self._tag_handler_reg):
                dpg.add_mouse_move_handler(callback=self.on_mouse_move)
                dpg.add_mouse_click_handler(callback=self.on_mouse_click)

    def _init_viewer(self, width: int, height: int):
        import threading
        from core.automation_manager import automation_manager
        if threading.current_thread() is threading.main_thread():
            self._init_viewer_internal(width, height)
        else:
            automation_manager.post_to_main_thread(self._init_viewer_internal, width, height, wait=True)

    def _init_viewer_internal(self, width: int, height: int):
        if dpg.does_item_exist(self._tag_texture): dpg.delete_item(self._tag_texture)
        if dpg.does_item_exist(self._tag_plot): dpg.delete_item(self._tag_plot)

        self.texture_width, self.texture_height = width, height
        
        dpg.add_raw_texture(width, height, default_value=self.convert_to_texture(np.zeros((height, width, 3), dtype=np.uint8)), 
                                tag=self._tag_texture, format=dpg.mvFormat_Float_rgb, parent=self._tag_texture_reg)
        
        plot_tag = dpg.add_plot(tag=self._tag_plot, no_menus=True, no_title=True, width=-1, height=-1, parent=self.winID)
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
        dpg.add_image_series(self._tag_texture, [0, 0], [width, height], parent=yaxis)
        dpg.fit_axis_data(xaxis)
        dpg.fit_axis_data(yaxis)

    def convert_to_texture(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return (rgb_frame.ravel().astype(np.float32)) / 255.0

    def _get_processed_rgb(self) -> np.ndarray | None:
        if self._raw_bgr_frame is None:
            return None
        
        bgr = self._raw_bgr_frame.copy()
        if self._process_enabled:
            # Saturation adjustment
            if abs(self._saturation - 1.0) > 1e-4:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                s = np.clip(s.astype(np.float32) * self._saturation, 0, 255).astype(np.uint8)
                hsv = cv2.merge([h, s, v])
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Gamma adjustment
            if abs(self._gamma - 1.0) > 1e-4:
                inv_gamma = 1.0 / max(0.01, self._gamma)
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                bgr = cv2.LUT(bgr, table)
        
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _on_proc_settings_change(self, *args, **kwargs):
        with self._lock:
            self._process_enabled = dpg.get_value(self._tag_proc_enable)
            self._saturation = dpg.get_value(self._tag_proc_sat)
            self._gamma = dpg.get_value(self._tag_proc_gamma)
            
            if self._raw_bgr_frame is not None:
                self._rgb_frame = self._get_processed_rgb()
        
        if self._raw_bgr_frame is not None:
            if getattr(self, "enabled", True):
                threading.Thread(target=self._set_image_worker, daemon=True).start()
            else:
                self._update_display()

    def input_cb(self, data=None, *args, **kwargs):
        if isinstance(data, dict):
            if data.get("clear", False) or data.get("reset_points", False):
                self.reset_points()
            if "enable" in data or "enabled" in data:
                was_enabled = getattr(self, "enabled", True)
                self.enabled = data.get("enable", data.get("enabled", True))
                if self.enabled and not was_enabled and getattr(self, "_rgb_frame", None) is not None:
                    threading.Thread(target=self._set_image_worker, daemon=True).start()
            return

        frame = data if data is not None else kwargs.get("data")
        if frame is None and args: frame = args[0]
            
        if isinstance(frame, tuple):
            frame = frame[0]
            
        if isinstance(frame, np.ndarray):
            h, w = frame.shape[:2]
            if h != self.texture_height or w != self.texture_width:
                self._init_viewer(w, h)
                
            with self._lock:
                self._frame = frame.copy()
                if len(frame.shape) == 2:
                    frame_8b = (frame / max(1, frame.max()) * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
                    self._raw_bgr_frame = cv2.cvtColor(frame_8b, cv2.COLOR_GRAY2BGR)
                else:
                    self._raw_bgr_frame = frame.copy() if frame.dtype == np.uint8 else (frame / max(1, frame.max()) * 255).astype(np.uint8)
                
                self._rgb_frame = self._get_processed_rgb()
            
            self.reset_points()
            if getattr(self, "enabled", True):
                threading.Thread(target=self._set_image_worker, daemon=True).start()
            else:
                self._update_display()

    def _set_image_worker(self):
        if not self._processing_lock.acquire(blocking=False): return
        try:
            if self.predictor is None or self._rgb_frame is None: return
            dpg.set_value(self._tag_status, "Encoding image...")
            dpg.configure_item(self._tag_status, color=(255, 255, 100))
            
            self.predictor.set_image(self._rgb_frame)
            
            has_points = False
            with self._lock:
                if self.points:
                    has_points = True
            
            if has_points:
                self._predict_worker_internal()
            else:
                dpg.set_value(self._tag_status, "Ready")
                dpg.configure_item(self._tag_status, color=(100, 255, 100))
                self._update_display()
        except Exception as e:
            logger.error(f"SAM error: {e}")
            dpg.set_value(self._tag_status, f"Error: {e}")
        finally:
            self._processing_lock.release()

    def reset_points(self, *args, **kwargs):
        with self._lock:
            self.points, self.labels, self._current_mask = [], [], None
        self._update_display()

    def on_mouse_move(self, sender, data, *args, **kwargs):
        try:
            if dpg.does_item_exist(self._tag_plot) and dpg.is_item_hovered(self._tag_plot):
                self.is_hovering = True
                mpos = dpg.get_plot_mouse_pos()
                self.mouse_x, self.mouse_y = int(mpos[0]), int(self.texture_height - mpos[1])
            else: self.is_hovering = False
        except Exception: self.is_hovering = False

    def on_mouse_click(self, sender, data, *args, **kwargs):
        if not getattr(self, "enabled", True): return
        if not self.is_hovering or not dpg.is_key_down(dpg.mvKey_LControl): return
        if self.mouse_x < 0 or self.mouse_x >= self.texture_width or self.mouse_y < 0 or self.mouse_y >= self.texture_height: return

        with self._lock:
            self.points.append([self.mouse_x, self.mouse_y])
            self.labels.append(1 if dpg.is_mouse_button_down(0) else 0)

        dpg.set_value(self._tag_status, "Predicting...")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        if not self._processing_lock.acquire(blocking=False): return
        try:
            self._predict_worker_internal()
        finally:
            self._processing_lock.release()

    def _predict_worker_internal(self):
        try:
            with self._lock:
                if self.predictor is None or not self.points: return
                pts, lbls = np.array(self.points), np.array(self.labels)

            masks, scores, _ = self.predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
            with self._lock: self._current_mask = masks[np.argmax(scores)]

            self._update_display()
            self._export_outputs()
            dpg.set_value(self._tag_status, "Ready. Mask updated.")
        except Exception as e:
            logger.error(f"SAM predict failed: {e}")

    def _update_display(self):
        with self._lock:
            if self._rgb_frame is None: return
            display_img = self._rgb_frame.copy()
            mask, pts, lbls = self._current_mask, self.points.copy(), self.labels.copy()

        if mask is not None:
            display_img[mask] = display_img[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            
        display_img_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        for (x, y), lbl in zip(pts, lbls):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(display_img_bgr, (x, y), radius=5, color=color, thickness=-1)
            cv2.circle(display_img_bgr, (x, y), radius=5, color=(255, 255, 255), thickness=1)

        dpg.set_value(self._tag_texture, self.convert_to_texture(display_img_bgr))

    def _export_outputs(self):
        with self._lock:
            mask, frame, pts = self._current_mask, self._rgb_frame, self.points
        if mask is None or frame is None or not pts: return
        mask_8u = (mask.astype(np.uint8) * 255)
        
        for module in self.connections.get("Mask", []):
            if hasattr(module, "input_cb"): module.input_cb(data=mask_8u, data_type=IOTypes.FRAME)
                
        overlay = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay[mask] = overlay[mask] * 0.5 + np.array([0, 0, 255]) * 0.5
        for module in self.connections.get("Overlay", []):
            if hasattr(module, "input_cb"): module.input_cb(data=overlay, data_type=IOTypes.FRAME)
                
        import uuid as uuid_mod
        x, y, w, h = cv2.boundingRect(mask_8u)
        sample = {
            "uuid": str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{self.UUID}_sam_roi")),
            "name": "SAM_Object", "mask": mask, "rect": [x, self.texture_height - (y + h), w, h], "action": "create"
        }
        for module in self.connections.get("ROI", []):
            if hasattr(module, "input_cb"): module.input_cb(sample=sample, data_type=IOTypes.ROI_SAMPLE)
        # ► 4. Emit Result Pair (Frame + Mask)
        result = (cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), mask_8u)
        for module in self.connections.get("Result", []):
            if hasattr(module, "input_cb"):
                module.input_cb(data=result, data_type=IOTypes.FRAME_MASK_PAIR)

EXPORTED_CLASS = SamInteractiveWin
EXPORTED_NAME = "SAM Interactive"
