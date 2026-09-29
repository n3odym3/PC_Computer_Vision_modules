"""
Color Mapper UI Window Module.

This module provides the DearPyGui interface for the Color Mapper module.
It enables interactive color mapping, palette management, interactive ROI
color calibration wizard, real-time ROI statistics export, and result
dispatching (e.g. to Color Aggregator or downstream plotting nodes).
"""

from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.window_base import WindowBase
from core.input_output_types import IOTypes
from loguru import logger
import numpy as np
import cv2
from pathlib import Path
from typing import Any
from .color_mapper_processor import ColorMapperProcessor
from config.display_scaling import display_scaling


class ColorMapperWin(WindowBase):
    """
    DearPyGui visual node for color classification and palette mapping.

    Integrates with `ColorMapperProcessor` to map pixels inside a mask to the
    closest reference color in a configurable palette. Provides an interactive image
    viewer, ROI probing tool, interactive palette editor, and data export pipelines.
    """

    folderpath: str | None
    groupname: str | None
    accepted_input_types: list[IOTypes]
    processor: ColorMapperProcessor
    roi_w: int
    roi_h: int
    mouse_x: int
    mouse_y: int
    is_hovering: bool
    selected_key: str | None
    _is_comparing: bool
    _selectable_tags: dict[str, str | int]
    _wizard_active: bool
    _wizard_index: int
    handler_tag: str
    item_handler_tag: str
    _persistent_fields: list[str]
    _tag_editor_win: str
    _tag_texture: str
    _tag_roi_rect: str
    _tag_roi_text: str
    _tag_roi_fill_mode: str
    _tag_roi_opacity: str
    _tag_new_key: str
    _tag_palette_list: str
    _tag_show_original: str
    _tag_compare_btn: str
    _plot_parent: str
    _tag_texture_reg: str
    _tag_status: str
    _tag_wizard_msg: str
    _tag_rt_out: str
    _tag_plot_node: str
    _tag_sample_name: str
    _tag_save_btn: str
    texture_width: int
    texture_height: int
    sub_windows: list[str]
    outputs: dict[str, IOTypes]
    connections: dict[str, list[Any]]
    
    def __init__(self, 
                 label: str = "Color Mapper",
                 win_width: int = 600,
                 win_height: int = 650,
                 pos: tuple[int, int] = (400, 50),
                 uuid: str | int | None = None,
                 outputs: dict[str, IOTypes] | None = None,
                 visible: bool = True) -> None:
        """
        Initialize the ColorMapper UI window, registries, and data processor.
        """
        outputs = outputs or {
            "Mapped Frame": IOTypes.FRAME,
            "Distribution": IOTypes.CMD_DICT,
            "CMD": IOTypes.CMD_DICT,
        }
        
        super().__init__(label=label, pos=pos,
                         win_width=win_width, win_height=win_height,
                         uuid=uuid, outputs=outputs, visible=visible)
        
        self.outputs = outputs
        self.connections = {k: [] for k in outputs}
        
        self.folderpath = None
        self.groupname = None

        # Accept Frame and Mask separately or together
        self.accepted_input_types = [
            IOTypes.FRAME,
            IOTypes.FRAME_MASK_PAIR,
            IOTypes.FOLDER_PATH,
            IOTypes.CMD_DICT,
            IOTypes.TEXT,
        ]
        
        # Processor integration
        config_path = Path(__file__).parent / "palette.json"
        self.processor = ColorMapperProcessor(config_path)
        
        # UI State
        self.roi_w = 20
        self.roi_h = 20
        self.mouse_x = 0
        self.mouse_y = 0
        self.is_hovering = False
        self.selected_key = None
        self._is_comparing = False
        
        keys = list(self.processor.color_map_RGB.keys())
        if keys:
            self.selected_key = keys[0]
        
        self._selectable_tags = {} 
        self._wizard_active = False
        self._wizard_index = 0
        self.handler_tag = f"mapper_handlers_{self.UUID}"
        self.item_handler_tag = f"mapper_item_handlers_{self.UUID}"
        self._persistent_fields = ["label"]
        
        # UI Tags
        self._tag_editor_win = f"mapper_editor_win_{self.UUID}"
        self._tag_texture = f"mapper_texture_{self.UUID}"
        self._tag_roi_rect = f"mapper_roi_rect_{self.UUID}"
        self._tag_roi_text = f"mapper_roi_text_{self.UUID}"
        self._tag_roi_fill_mode = f"mapper_roi_fill_{self.UUID}"
        self._tag_roi_opacity = f"mapper_roi_opacity_{self.UUID}"
        self._tag_new_key = f"mapper_new_key_{self.UUID}"
        self._tag_palette_list = f"mapper_pal_list_{self.UUID}"
        self._tag_show_original = f"mapper_show_orig_{self.UUID}"
        self._tag_compare_btn = f"mapper_compare_{self.UUID}"
        self._plot_parent = f"plot_parent_{self.UUID}"
        self._tag_texture_reg = f"mapper_texture_reg_{self.UUID}"

        dpg.add_texture_registry(tag=self._tag_texture_reg, show=False)
        
        self._tag_status = f"mapper_status_{self.UUID}"
        self._tag_wizard_msg = f"mapper_wizard_msg_{self.UUID}"
        self._tag_rt_out = f"mapper_rt_out_{self.UUID}"
        self._tag_plot_node = f"mapper_plot_{self.UUID}"
        self._tag_sample_name = f"mapper_sample_name_{self.UUID}"
        self._tag_save_btn = f"mapper_save_btn_{self.UUID}"
        
        self.texture_width = 100
        self.texture_height = 100
        
        # Register sub-windows for NodeEditor sync (colorization, double-click)
        self.sub_windows = [self._tag_editor_win]
        
        self._build_ui()

        self.update_permission()

    def update_permission(self) -> None:
        """
        Adjust module permissions and UI visibility based on `app_state.mode`.

        Disables closing the window when running in user mode.
        """
        from core.app_state import app_state
        is_user = (app_state.mode == "user")
        
        if dpg.does_item_exist(self.winID):
            dpg.configure_item(self.winID, no_close=is_user)

    def _confirm_save_palette(self, sender: str | int | None = None, app_data: Any = None, user_data: Any = None, *args, **kwargs) -> None:
        """
        Display a modal dialog asking for user confirmation before saving the palette to disk.
        """
        s = display_scaling.scale
        if dpg.does_item_exist("cfg_save_modal"): dpg.delete_item("cfg_save_modal")
        with dpg.window(label="Confirm Save", modal=True, show=True, tag="cfg_save_modal", no_title_bar=False, pos=(self.pos[0]+50, self.pos[1]+50)):
            dpg.add_text("Overwrite existing palette configuration?\nThis action is irreversible.")
            dpg.add_separator()
            btn_w = s(100)
            with dpg.group(horizontal=True):
                dpg.add_button(label="OK", width=btn_w, callback=lambda: [self.processor.save_palette(), dpg.delete_item("cfg_save_modal")])
                dpg.add_button(label="Cancel", width=btn_w, callback=lambda: dpg.delete_item("cfg_save_modal"))

    def _request_redraw(self, sender: str | int | None = None, app_data: Any = None, user_data: Any = None, *args, **kwargs) -> None:
        """
        Trigger a re-render/re-evaluation cycle if a frame is loaded.
        """
        if self.processor.frame is not None:
            self._process_all()

    def _build_ui(self) -> None:
        """
        Construct the DearPyGui main window layout, controls, plot view, and event handlers.
        """
        with dpg.window(label=self.label,
                        width=self.win_width, height=self.win_height,
                        pos=self.pos, tag=self.winID, show=self.visible):
            
            s = display_scaling.scale
            
            # -- Row 1: Main Actions & View Modes --
            with dpg.group(horizontal=True):
                dpg.add_button(label="Process All Frame", tag=f"mapper_process_all_{self.UUID}", callback=self._process_all_frame_cb)
                dpg.add_checkbox(label="Show original", tag=self._tag_show_original, callback=self._request_redraw)
                dpg.add_button(label="Compare", tag=self._tag_compare_btn, width=s(150))

            # -- Row 2: ROI & Settings --
            with dpg.group(horizontal=True):
                dpg.add_text("ROI:", color=(100, 200, 100))
                dpg.add_drag_int(default_value=self.roi_w, width=s(40), min_value=1, tag=f"mapper_roi_w_{self.UUID}", callback=lambda s, a: setattr(self, 'roi_w', a))
                dpg.add_text("x")
                dpg.add_drag_int(default_value=self.roi_h, width=s(40), min_value=1, tag=f"mapper_roi_h_{self.UUID}", callback=lambda s, a: setattr(self, 'roi_h', a))
                
                dpg.add_combo(["No Fill", "Mean RGB", "Closest Color"], default_value="Closest Color", tag=self._tag_roi_fill_mode, width=s(150))
                dpg.add_text("A:")
                dpg.add_drag_int(default_value=120, min_value=0, max_value=255, tag=self._tag_roi_opacity, width=s(100))
                
                dpg.add_checkbox(label="RT Out", tag=self._tag_rt_out)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Export ROI distribution in real-time")
                    
                dpg.add_button(label="Edit Palette", tag=f"mapper_edit_palette_{self.UUID}", callback=self._toggle_editor, width=-1)

            with dpg.group(horizontal=True):
                dpg.add_text("Status:", color=(150, 150, 150))
                dpg.add_text("Wait...", tag=self._tag_status, color=(200, 200, 100))

            # -- Row 3: Save to Color Agregator --
            with dpg.group(horizontal=True):
                dpg.add_text("Sample name/ID:", color=(150, 200, 255))
                dpg.add_input_text(hint="e.g. Sample_01", tag=self._tag_sample_name, width=180)
                dpg.add_button(label="Save to Agregator", tag=self._tag_save_btn,
                               callback=self._save_to_agregator_cb)
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("Sends current distribution with save=True\nso the Color Agregator commits a new row.")

            dpg.add_separator()
            
            # -- Image Viewer --
            with dpg.child_window(tag=self._plot_parent, height=-1):
                pass 
                        
            # Handlers
            self.handler_tag = f"mapper_handlers_{self.UUID}"
            with dpg.handler_registry(tag=self.handler_tag):
                dpg.add_mouse_move_handler(callback=self._on_mouse_move)
                dpg.add_mouse_click_handler(callback=self._on_mouse_click)
                dpg.add_mouse_release_handler(callback=self._compare_off)
            
            self.item_handler_tag = f"mapper_comp_hand_{self.UUID}"
            if not dpg.does_item_exist(self.item_handler_tag):
                with dpg.item_handler_registry(tag=self.item_handler_tag):
                    dpg.add_item_clicked_handler(callback=self._compare_on)
            dpg.bind_item_handler_registry(self._tag_compare_btn, self.item_handler_tag)

            self._init_viewer(400, 300)
            
        # -- Secondary Window (Palette Editor) --
        with dpg.window(label=f"Palette Editor & Calibration [{self.label}]", 
                        width=s(300), height=s(400), show=False, 
                        tag=self._tag_editor_win, pos=(self.pos[0]+60, self.pos[1]+60)):
                        
            # -- Row 0: Instructions --
            dpg.add_text("CTRL+Click on image to calibrate", tag=self._tag_wizard_msg, color=(255, 255, 0)) 
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", tag=f"mapper_editor_save_{self.UUID}", callback=self._confirm_save_palette)
                dpg.add_button(label="Clear", tag=f"mapper_editor_clear_{self.UUID}", callback=self._clear_all_palette)
                dpg.add_button(label="Calibration", callback=self._start_wizard, tag=f"wizard_btn_{self.UUID}")

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint="New Key ...", tag=self._tag_new_key, width=-s(100))
                dpg.add_button(label="Add", tag=f"mapper_editor_add_{self.UUID}", width=-1, callback=self._add_key)
                
            dpg.add_separator()
            
            with dpg.child_window(tag=self._tag_palette_list, height=-1):
                pass
                
        self._build_palette_list()

    def is_ready(self) -> bool:
        """Indicate whether the module is ready to accept inputs."""
        return True

    def close(self) -> None:
        """
        Clean up resources, handler registries, and memory allocations upon window closure.
        """
        if dpg.does_item_exist(self.item_handler_tag):
            dpg.delete_item(self.item_handler_tag)
        if dpg.does_item_exist(self.handler_tag):
            dpg.delete_item(self.handler_tag)
        if hasattr(self, 'processor') and self.processor:
            self.processor.reset_inputs()
        super().close()

    def on_close(self) -> None:
        """Legacy close handler alias."""
        self.close()

    def _toggle_editor(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Open and focus the floating Palette Editor and Calibration secondary window.
        """
        if dpg.does_item_exist(self._tag_editor_win):
            dpg.configure_item(self._tag_editor_win, show=True)
            dpg.focus_item(self._tag_editor_win)

    def _process_all_frame_cb(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Callback to create a full-frame mask and run color mapping across all pixels.
        """
        if self.processor.set_full_mask():
            self._process_all()
        else:
            logger.warning("ColorMapper: Cannot process all frame without a frame input.")

    def _compare_on(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Callback when Compare button is pressed to switch viewer to masked original frame preview.
        """
        self._is_comparing = True
        self._request_redraw()

    def _compare_off(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Callback when mouse is released to exit Compare preview mode.
        """
        if self._is_comparing:
            self._is_comparing = False
            self._request_redraw()

    def _add_key(self, *args, **kwargs) -> None:
        """
        Callback to add a new color entry from the text input to the palette.
        """
        new_key = dpg.get_value(self._tag_new_key)
        if self.processor.add_key(new_key):
            self.selected_key = new_key.strip()
            self._build_palette_list()
            dpg.set_value(self._tag_new_key, "")

    def _select_key_cb(self, sender: str | int | None = None, app_data: Any = None, user_data: str | None = None, *args, **kwargs) -> None:
        """
        Callback when a palette item is selected in the palette editor list.
        """
        self.selected_key = user_data
        # Full rebuild to refresh the selection indicators
        self._build_palette_list()

    def _delete_key_cb(self, sender: str | int | None = None, app_data: Any = None, user_data: str | None = None, *args, **kwargs) -> None:
        """
        Callback to remove a color key from the palette.
        """
        if self.processor.delete_key(user_data):
            if self.selected_key == user_data:
                self.selected_key = None
            self._build_palette_list()

    def _rename_key_confirmed(self, old_name: str, new_name: str) -> None:
        """
        Apply key renaming in processor and refresh palette list UI.
        """
        if self.processor.rename_key(old_name, new_name):
            if self.selected_key == old_name:
                self.selected_key = new_name.strip()
            self._build_palette_list()
        if dpg.does_item_exist("rename_modal"): dpg.delete_item("rename_modal")

    def _open_rename_modal(self, sender: str | int | None = None, app_data: Any = None, user_data: str | None = None, *args, **kwargs) -> None:
        """
        Open modal dialog to rename a selected palette key.
        """
        old_name = user_data
        s = display_scaling.scale
        if dpg.does_item_exist("rename_modal"): dpg.delete_item("rename_modal")
        
        with dpg.window(label=f"Rename '{old_name}'", modal=True, show=True, tag="rename_modal", no_title_bar=False, pos=(self.pos[0]+50, self.pos[1]+50)):
            dpg.add_text(f"New name for '{old_name}':")
            input_tag = dpg.add_input_text(default_value=old_name, on_enter=True, width=-1)
            dpg.add_separator()
            btn_w = s(100)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Rename", width=btn_w, callback=lambda: self._rename_key_confirmed(old_name, dpg.get_value(input_tag)))
                dpg.add_button(label="Cancel", width=btn_w, callback=lambda: dpg.delete_item("rename_modal"))

    def _edit_color_cb(self, sender: str | int | None, app_data: list[float], user_data: tuple[str, str | int], *args, **kwargs) -> None:
        """
        Callback triggered when a color is modified using the inline DearPyGui color picker.
        """
        name, label_tag = user_data
        # DPG add_color_edit app_data is a list of floats from 0.0 to 255.0
        # Determine format (handle if it's 0-1 or 0-255)
        if max(app_data[:3]) <= 1.0 and sum(app_data[:3]) > 0:
            r, g, b = int(app_data[0]*255), int(app_data[1]*255), int(app_data[2]*255)
        else:
            r, g, b = int(app_data[0]), int(app_data[1]), int(app_data[2])
            
        new_rgb = (r, g, b)
        
        self.processor.color_map_RGB[name] = new_rgb
        self.processor.precompute_palette()
        
        # Update UI label directly instead of rebuilding the list
        if dpg.does_item_exist(label_tag):
            dpg.set_item_label(label_tag, f"{name} ({r},{g},{b})")

    def _build_palette_list(self) -> None:
        """
        Reconstruct the palette list UI items, color pastilles, and context menus.
        """
        dpg.delete_item(self._tag_palette_list, children_only=True)
        s = display_scaling.scale
        keys = self.processor.palette_keys
        
        if self.selected_key not in keys and keys:
            self.selected_key = keys[0]
            
        self._selectable_tags = {}
        for name, rgb in self.processor.color_map_RGB.items():
            is_selected = (name == self.selected_key)
            label = f"{name} ({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
            
            with dpg.group(horizontal=True, parent=self._tag_palette_list):
                sel_tag = dpg.generate_uuid()
                self._selectable_tags[name] = sel_tag
                
                # Interactive pastille (Color Picker on click)
                col_rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 255.0]
                dpg.add_color_edit(default_value=col_rgba, no_inputs=True, no_label=True, 
                                    width=s(18), height=s(18),
                                    user_data=(name, sel_tag), callback=self._edit_color_cb)
                
                dpg.add_selectable(label=label, 
                                    default_value=is_selected,
                                    callback=self._select_key_cb, 
                                    user_data=name, tag=sel_tag)

                with dpg.popup(sel_tag, mousebutton=dpg.mvMouseButton_Right):
                    dpg.add_text(f"Options: {name}", color=(200, 200, 100))
                    dpg.add_separator()
                    
                    dpg.add_selectable(label="Set Color from ROI", 
                                       callback=lambda s, a, u: [setattr(self, 'selected_key', u), self._calibrate_roi()], 
                                       user_data=name)
                    
                    dpg.add_separator()
                    dpg.add_selectable(label="Rename Key", callback=self._open_rename_modal, user_data=name)
                    dpg.add_selectable(label="Delete Key", callback=self._delete_key_cb, user_data=name)

    def _init_viewer(self, width: int, height: int) -> None:
        """
        Thread-safe wrapper to initialize or resize the viewer texture and plot.
        """
        import threading
        from core.automation_manager import automation_manager
        if threading.current_thread() is threading.main_thread():
            self._init_viewer_internal(width, height)
        else:
            automation_manager.post_to_main_thread(self._init_viewer_internal, width, height, wait=True)

    def _init_viewer_internal(self, width: int, height: int) -> None:
        """
        Create or recreate the raw texture and plot axes for image rendering.
        """
        if dpg.does_item_exist(self._tag_texture): dpg.delete_item(self._tag_texture)
        if dpg.does_item_exist(self._tag_plot_node): dpg.delete_item(self._tag_plot_node)
        
        self.texture_width = width
        self.texture_height = height
        
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        dpg.add_dynamic_texture(width, height, self._convert_to_dpg(img), tag=self._tag_texture, parent=self._tag_texture_reg)
        
        plot_tag = dpg.add_plot(tag=self._tag_plot_node, no_menus=True, no_title=True, width=-1, height=-1, parent=self._plot_parent)
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, no_gridlines=True, no_tick_labels=True, no_tick_marks=True, parent=plot_tag)
        dpg.add_image_series(self._tag_texture, [0, 0], [width, height], parent=yaxis)
        
        dpg.draw_rectangle(pmin=(0,0), pmax=(0,0), color=(0, 255, 0, 255), thickness=2, tag=self._tag_roi_rect, parent=plot_tag, show=False)
        dpg.add_plot_annotation(label="ROI", default_value=(0,0), color=(0, 255, 0, 255), offset=(10,-10), tag=self._tag_roi_text, parent=plot_tag, show=False)

    def _convert_to_dpg(self, img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR/BGRA numpy image to flattened float32 RGBA format expected by DearPyGui raw textures.
        """
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        return (rgba.ravel().astype(np.float32)) / 255.0

    def _on_mouse_move(self, sender: str | int | None, data: Any, *args, **kwargs) -> None:
        """
        Mouse move event handler to track plot cursor coordinates and update ROI overlay.
        """
        try:
            if dpg.does_item_exist(self._tag_plot_node) and dpg.is_item_hovered(self._tag_plot_node):
                self.is_hovering = True
                mpos = dpg.get_plot_mouse_pos()
                self.mouse_x = int(mpos[0])
                self.mouse_y = int(mpos[1])
                self._update_roi_overlay()
            else:
                self.is_hovering = False
                self._update_roi_overlay()
        except Exception:
            self.is_hovering = False

    def _update_roi_overlay(self) -> None:
        """
        Update the position, fill color, text annotation, and real-time output of the inspection ROI.
        """
        if not hasattr(self, "_tag_roi_rect") or not dpg.does_item_exist(self._tag_roi_rect):
            return
            
        if not self.is_hovering:
            dpg.configure_item(self._tag_roi_rect, show=False)
            dpg.configure_item(self._tag_roi_text, show=False)
            return

        w, h = self.roi_w, self.roi_h
        x = self.mouse_x - w // 2
        y = self.mouse_y - h // 2
        
        img_w, img_h = self.texture_width, self.texture_height
        if img_w == 0 or img_h == 0: return

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        if x2 > x1 and y2 > y1:
            y1_arr = max(0, min(img_h - y2, img_h))
            y2_arr = max(0, min(img_h - y1, img_h))
            
            fill_color = (0, 0, 0, 0)
            text_str = ""
            
            use_mask = not dpg.get_value(self._tag_show_original)
            # Fallback: if no mask is loaded, we can't use it for ROI analysis
            actual_use_mask = use_mask and self.processor.mask is not None
            
            mean_rgb, closest_name = self.processor.analyze_roi(x1, y1_arr, x2, y2_arr, actual_use_mask)
            
            # Real-time export of ROI proportions for external modules (e.g. Circular Plot)
            if dpg.get_value(self._tag_rt_out):
                try:
                    roi_dist = self.processor.get_roi_distribution(x1, y1_arr, x2, y2_arr, actual_use_mask)
                    if roi_dist:
                        self._emit_distribution(roi_dist, title_suffix=f"(ROI {self.UUID})")
                except Exception as e:
                    logger.error(f"ColorMapper: Failed to export ROI distribution: {e}")

            if mean_rgb is not None:
                text_str = f"RGB: {mean_rgb}\nGroup: {closest_name}"
                
                fill_mode = dpg.get_value(self._tag_roi_fill_mode)
                opacity = dpg.get_value(self._tag_roi_opacity)
                
                if fill_mode == "Mean RGB":
                    fill_color = (*mean_rgb, opacity)
                elif fill_mode == "Closest Color":
                    c_rgb = self.processor.color_map_RGB.get(closest_name, (0,0,0))
                    fill_color = (int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2]), opacity)

            dpg.configure_item(self._tag_roi_rect, pmin=(x1, y1), pmax=(x2, y2), fill=fill_color, show=True)
            dpg.configure_item(self._tag_roi_text, default_value=(x2, y2), label=text_str, show=bool(text_str))

        else:
            dpg.configure_item(self._tag_roi_rect, show=False)
            dpg.configure_item(self._tag_roi_text, show=False)

    def _on_mouse_click(self, sender: str | int | None, data: Any, *args, **kwargs) -> None:
        """
        Handle mouse click events to trigger ROI calibration when CTRL key is held.
        """
        if dpg.is_key_down(dpg.mvKey_LControl) and self.is_hovering:
            self._calibrate_roi()
            
    def _clear_all_palette(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Clear all entries from the palette.
        """
        self.processor.color_map_RGB = {}
        self.processor.precompute_palette()
        self.selected_key = None
        self._wizard_active = False
        self._build_palette_list()

    def _start_wizard(self, sender: str | int | None = None, app_data: Any = None, *args, **kwargs) -> None:
        """
        Toggle interactive step-by-step calibration wizard for all keys in the palette.
        """
        # Toggle logic
        if self._wizard_active:
            self._wizard_active = False
            dpg.configure_item(f"wizard_btn_{self.UUID}", label="Calibration")
            dpg.set_value(self._tag_wizard_msg, "CTRL+Click on image to calibrate")
            return

        keys = self.processor.palette_keys
        if not keys:
            logger.warning("ColorMapper: No keys in palette to recalibrate.")
            return
        
        self._wizard_active = True
        self._wizard_index = 0
        self.selected_key = keys[0]
        self._build_palette_list() 
        dpg.configure_item(f"wizard_btn_{self.UUID}", label="STOP Calibration")
        dpg.set_value(self._tag_wizard_msg, f"CALIBRATION: Please CTRL+Click on '{self.selected_key}'")
        logger.info(f"ColorMapper: Calibration started. Please calibrate '{self.selected_key}'")

    def _calibrate_roi(self) -> None:
        """
        Sample the mean color under the current ROI and calibrate the selected palette key.
        """
        if self.selected_key is None:
            return

        w, h = self.roi_w, self.roi_h
        x = self.mouse_x - w // 2
        y = self.mouse_y - h // 2
        
        img_h = self.texture_height
        img_w = self.texture_width
        
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        y1_arr = max(0, min(img_h - y2, img_h))
        y2_arr = max(0, min(img_h - y1, img_h))
        
        use_mask = not dpg.get_value(self._tag_show_original)
        if self.processor.calibrate_key_from_roi(self.selected_key, x1, y1_arr, x2, y2_arr, use_mask):
            logger.info(f"ColorMapper: Calibrated '{self.selected_key}'")
            
            if self._wizard_active:
                keys = self.processor.palette_keys
                self._wizard_index += 1
                if self._wizard_index < len(keys):
                    self.selected_key = keys[self._wizard_index]
                    dpg.set_value(self._tag_wizard_msg, f"CALIBRATION: Please CTRL+Click on '{self.selected_key}'")
                else:
                    self._wizard_active = False
                    dpg.configure_item(f"wizard_btn_{self.UUID}", label="Calibration")
                    dpg.set_value(self._tag_wizard_msg, "Calibration Finished! (CTRL+Click to calibrate)")
            
            self._build_palette_list()

    def _emit_distribution(self, dist: dict[str, dict], title_suffix: str = "", save: bool = False, sample_name: str = "") -> None:
        """
        Utility to send distribution data payload to all connected downstream modules.
        """
        pie_data = {
            "action": "plot_pie",
            "title": f"Color Distribution {title_suffix}",
            "distribution": dist,
            "save": save,
            "sample_name": sample_name,
        }
        for module in self.connections.get("Distribution", []):
            module.input_cb(data=pie_data, data_type=IOTypes.CMD_DICT)

    def _save_to_agregator_cb(self, *args, **kwargs) -> None:
        """
        Save the current distribution to the Color Aggregator, crop masked images, and clear points.
        """
        if not hasattr(self.processor, 'last_distribution') or not self.processor.last_distribution:
            logger.warning("ColorMapper: No distribution available yet – process a frame first.")
            return
        name = ""
        if dpg.does_item_exist(self._tag_sample_name):
            name = dpg.get_value(self._tag_sample_name).strip()
            
        if not name:
            try:
                from core.tutorial_manager import tutorial_manager
                tutorial_manager.play_error_hint(self._tag_sample_name, "Please provide a Sample name/ID before exporting.")
            except ImportError:
                logger.warning("ColorMapper: Sample name is empty!")
            return

        base_name = getattr(self, 'groupname', None)
        if not base_name:
            base_name = name if name else "Image"
            sample_str = ""
        else:
            sample_str = f"_{name}" if name else ""

        missing_requirements = []
        if not getattr(self, 'folderpath', None): missing_requirements.append("folderpath")
        if self.processor.mask is None: missing_requirements.append("mask")
        if self.processor.frame is None: missing_requirements.append("frame")

        if not missing_requirements:
            try:
                import os
                mask = self.processor.mask
                if mask.shape[:2] != self.processor.frame.shape[:2]:
                    mask = cv2.resize(mask, (self.processor.frame.shape[1], self.processor.frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid_contours = [c for c in contours if cv2.contourArea(c) > 0]
                
                if valid_contours:
                    all_pts = np.vstack(valid_contours)
                    x, y, w, h = cv2.boundingRect(all_pts)
                    
                    raw_frame = self.processor.frame
                    mapped_frame = getattr(self.processor, '_cached_mapped_frame', None)
                    if mapped_frame is None:
                        mapped_frame = raw_frame
                        
                    def crop_and_transparent(img_bgr, roi_mask, cx, cy, cw, ch):
                        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                            img_bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
                        else:
                            img_bgra = img_bgr.copy()
                        cropped = img_bgra[cy:cy+ch, cx:cx+cw].copy()
                        c_mask = roi_mask[cy:cy+ch, cx:cx+cw]
                        # Set fully black (0,0,0) AND fully transparent for pixels outside mask
                        cropped[c_mask <= 127] = [0, 0, 0, 0]
                        return cropped

                    raw_cropped = crop_and_transparent(raw_frame, mask, x, y, w, h)
                    mapped_cropped = crop_and_transparent(mapped_frame, mask, x, y, w, h)
                    
                    raw_name = f"{base_name}{sample_str}_RAW.png"
                    proc_name = f"{base_name}{sample_str}_processed.png"
                    
                    raw_path = os.path.join(self.folderpath, raw_name)
                    proc_path = os.path.join(self.folderpath, proc_name)
                    
                    cv2.imwrite(raw_path, raw_cropped)
                    cv2.imwrite(proc_path, mapped_cropped)
                    logger.info(f"ColorMapper: Saved cropped images to {self.folderpath}")
                else:
                    logger.warning("ColorMapper: Mask has no valid contours, cannot crop.")
            except Exception as e:
                logger.error(f"ColorMapper: Failed to save cropped images: {e}")
        else:
            logger.warning(f"ColorMapper: Cropped images NOT saved. Missing: {', '.join(missing_requirements)}")

        self._emit_distribution(
            self.processor.last_distribution,
            title_suffix="(Saved)",
            save=True,
            sample_name=name,
        )
        if dpg.does_item_exist(self._tag_sample_name):
            dpg.set_value(self._tag_sample_name, "")
        logger.success(f"ColorMapper: Saved distribution for '{name or 'unnamed'}' to Color Agregator.")

        # Emit clear command to clear points in SAM
        for module in self.connections.get("CMD", []):
            if hasattr(module, 'input_cb'):
                module.input_cb(data={"clear": True, "reset_points": True}, data_type=IOTypes.CMD_DICT)

    def input_cb(self, data: Any, *args, **kwargs) -> None:
        """
        Entry point receiving data from upstream nodes (Frames, Masks, Folder Paths, Commands).
        """
        data_type = kwargs.get("data_type")
        if data_type == IOTypes.FOLDER_PATH:
            self.folderpath = data
            return
        if data_type == IOTypes.TEXT and isinstance(data, str):
            self.groupname = data[:-4] if data.lower().endswith(".png") else data
            return
        if data_type == IOTypes.CMD_DICT and isinstance(data, dict):
            if "full_name" in data:
                fn = data["full_name"]
                self.groupname = fn[:-4] if fn.lower().endswith(".png") else fn
            if "clear" in data and data.get("clear", False):
                self.folderpath = None
                self.groupname = None
            return

        # Handle metadata wrapper (dict, array/tuple) safely accommodating tuples formatted like (frame, 'Alvium')
        if isinstance(data, tuple) and len(data) >= 2:
            if isinstance(data[0], dict) or isinstance(data[0], str):
                data = data[1]
            elif isinstance(data[0], np.ndarray) and not isinstance(data[1], np.ndarray):
                data = data[0]

        data_type = kwargs.get("data_type")
        is_pair = data_type == IOTypes.FRAME_MASK_PAIR or (isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], np.ndarray) and isinstance(data[1], np.ndarray))
        
        if self.processor.ingest_data(data, is_pair=is_pair):
            status = []
            if self.processor.frame is not None: status.append("Frame [OK]")
            if self.processor.mask is not None: status.append("Mask [OK]")
            # Always call _process_all so the viewer updates with the latest frame.
            # The processor internally decides if it needs heavy mapping or not.
            self._process_all()

    def _process_all(self) -> None:
        """
        Request color mapping execution from processor, update UI texture and status,
        and dispatch outputs to downstream connected modules.
        """
        try:
            dpg.set_value(self._tag_status, "Processing Map...")
            dpg.configure_item(self._tag_status, color=(255, 255, 100))
            
            show_orig = dpg.get_value(self._tag_show_original)
            
            output_bgr, distribution, msg = self.processor.process_map(show_original=show_orig, is_comparing=self._is_comparing)
            if output_bgr is None:
                return
                
            dpg.set_value(self._tag_status, msg)
            if msg.startswith("Ready"):
                dpg.configure_item(self._tag_status, color=(100, 255, 100))
            
            # Output Display
            if output_bgr.shape[0] != self.texture_height or output_bgr.shape[1] != self.texture_width:
                self._init_viewer(output_bgr.shape[1], output_bgr.shape[0])
            
            dpg.set_value(self._tag_texture, self._convert_to_dpg(output_bgr))
            
            # Emit Frame
            for module in self.connections.get("Mapped Frame", []):
                module.input_cb(data=output_bgr, data_type=IOTypes.FRAME)
                
            # Cache distribution for later Save action
            self.processor.last_distribution = distribution

            # Emit CMD_DICT for the Plotter (Only if not bypassed)
            if not dpg.get_value(self._tag_show_original):
                self._emit_distribution(distribution, title_suffix=f"(Full Frame {self.UUID})")
            
        except Exception as e:
            logger.error(f"ColorMapper Error: {e}")
            dpg.set_value(self._tag_status, f"Error: {e}")
            dpg.configure_item(self._tag_status, color=(255, 100, 100))

EXPORTED_CLASS = ColorMapperWin
EXPORTED_NAME = "Color Mapper"
