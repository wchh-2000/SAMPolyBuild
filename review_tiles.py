"""Lightweight Matplotlib viewer for reviewing canonical per-tile polygons."""

import argparse
import sys
import time
from pathlib import Path

from utils.image_tile_reader import ImageTileReader
from workflow.auto_bbox_provider import select_bbox_for_click
from workflow.manifest import TileManifest
from workflow.prompt_infer_adapter import (
    PromptSessionCache,
    make_add_bbox,
    run_mock_prompt_inference,
    run_prompt_inference_on_tile,
)
from workflow.prompt_review_session import PromptReviewSession
from workflow.refine_polygon_validation import RefinePolygonValidationConfig
from workflow.refine_session import RefineSession
from workflow.tile_review import (
    add_click_polygon,
    first_review_index,
    load_auto_result,
    load_review_result,
    next_source_polygon_id,
    resolve_review_paths,
    resolve_tile_image,
    save_review_result,
    select_polygon_at_point,
    set_polygon_status,
)


HELP_TEXT = """Tile review shortcuts:
  N / Right Arrow  save and open next tile
  P / Left Arrow   save and open previous tile
  Left click       in Prompt mode: start a building or add a positive point
  Right click      in Prompt/Refine mode: add a negative point to the running building
  Enter            accept the running Prompt/Refine session
  C / Esc          cancel the running Prompt session
  A                enter Add mode for a missed building
  R                refine the selected polygon
  D                mark selected polygon deleted
  U                restore selected polygon to active
  S                save current tile
  Esc              cancel Add/Prompt/Refine mode, otherwise save and quit
  Q                save and quit
  H                print this help
"""


def navigation_step_for_key(key) -> int | None:
    """Map standard and Linux keypad key names to one tile-navigation step."""
    normalized = str(key or "").strip().lower()
    if normalized in ("n", "right", "6", "kp_6"):
        return 1
    if normalized in ("p", "left", "4", "kp_4"):
        return -1
    return None


def is_auto_repeat_key_event(event) -> bool:
    """Read Qt's native auto-repeat flag without requiring a Qt import."""
    native_event = getattr(event, "guiEvent", None)
    is_auto_repeat = getattr(native_event, "isAutoRepeat", None)
    return bool(is_auto_repeat()) if callable(is_auto_repeat) else False


def parse_args() -> argparse.Namespace:
    """Parse paths, resume position, and polygon selection distance."""
    parser = argparse.ArgumentParser(description="Review Auto polygons one tile at a time.")
    parser.add_argument("--manifest", required=True, help="Canonical manifest.json path.")
    parser.add_argument("--auto-results", default=None, help="Per-tile Auto result directory.")
    parser.add_argument("--edited-results", default=None, help="Reviewed result directory.")
    parser.add_argument("--start-tile", default=None, help="Optional tile_id to open first.")
    parser.add_argument(
        "--selection-distance", type=float, default=10.0,
        help="Maximum pixel distance for nearest-boundary selection.",
    )
    parser.add_argument("--config", default=None, help="Prompt JSON config path for Add mode.")
    parser.add_argument("--checkpoint", default=None, help="Prompt checkpoint path for Add mode.")
    parser.add_argument("--device", default="cuda", help="Prompt device: cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--default-mode",
        choices=("prompt", "select"),
        default="prompt",
        help="Prompt-first mode or the legacy select-first mode.",
    )
    parser.add_argument(
        "--bbox-provider",
        choices=("auto", "fixed", "auto_or_fixed"),
        default="auto_or_fixed",
        help="How Prompt-first mode selects the starting bbox for one building.",
    )
    parser.add_argument(
        "--bbox-nearest-threshold", type=float, default=80.0,
        help="Maximum pixel distance for auto-nearest bbox fallback.",
    )
    parser.add_argument(
        "--prompt-source-name", default="prompt",
        help="Source field used when Prompt-first mode saves a confirmed polygon.",
    )
    parser.add_argument(
        "--show-auto",
        dest="show_auto",
        action="store_true",
        help="Draw Auto polygons for debugging instead of hiding them.",
    )
    parser.add_argument(
        "--hide-auto",
        dest="show_auto",
        action="store_false",
        help="Hide Auto polygons while still using them as Prompt bbox candidates.",
    )
    parser.set_defaults(show_auto=False)
    parser.add_argument(
        "--add-bbox-size", type=int, default=256,
        help="Centered bbox side length used for an Add-mode click.",
    )
    parser.add_argument(
        "--mock-prompt", action="store_true",
        help="Generate a deterministic rectangle without loading a model.",
    )
    parser.add_argument(
        "--debug-prompt-points", action="store_true",
        help="Print prompt point coordinates and labels through the review prediction chain.",
    )
    parser.add_argument(
        "--refine-by-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable point-guided mask refinement before polygon extraction.",
    )
    parser.add_argument(
        "--point-refine-mode",
        choices=("connected_component", "distance"),
        default="distance",
        help="Point-guided refinement mode used when --refine-by-points is enabled.",
    )
    parser.add_argument(
        "--negative-feature-refine", action="store_true",
        help="Use negative-point feature prototypes to remove visually similar mask pixels.",
    )
    parser.add_argument(
        "--negative-similarity-thr", type=float, default=0.75,
        help="Similarity threshold for negative feature refinement.",
    )
    parser.add_argument(
        "--negative-feature-patch-size", type=int, default=21,
        help="Patch size around each negative point for feature prototype extraction.",
    )
    parser.add_argument(
        "--negative-protect-radius", type=int, default=20,
        help="Radius around positive points protected from negative feature refinement.",
    )
    parser.add_argument(
        "--negative-spatial-sigma", type=float, default=80.0,
        help="Spatial falloff sigma for negative feature refinement. Use <=0 to disable weighting.",
    )
    parser.add_argument(
        "--debug-refine-dir", default=None,
        help="Optional debug output directory. Defaults to work_dir/debug_refine when debugging is enabled.",
    )
    parser.add_argument(
        "--allow-outside-negative-points",
        action="store_true",
        help="Allow right-click negative points outside the current preview polygon or mask.",
    )
    parser.add_argument(
        "--negative-point-tolerance", type=float, default=5.0,
        help="Allow negative points up to this many pixels outside the preview polygon boundary.",
    )
    parser.add_argument(
        "--max-polygon-points", type=int, default=80,
        help="Reject refined polygons whose simplified point count exceeds this threshold.",
    )
    parser.add_argument(
        "--min-polygon-area", type=float, default=50.0,
        help="Reject refined polygons smaller than this area.",
    )
    parser.add_argument(
        "--max-area-change-ratio", type=float, default=1.5,
        help="Reject refined polygons whose area grows above this ratio of the previous preview.",
    )
    parser.add_argument(
        "--min-area-change-ratio", type=float, default=0.3,
        help="Reject refined polygons whose area drops below this ratio of the previous preview.",
    )
    parser.add_argument(
        "--reject-invalid-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject invalid refined polygons and keep the previous preview.",
    )
    return parser.parse_args()


def display_image(image, np):
    """Convert one tile to an RGB uint8 display image without changing source data."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 1:
        raise ValueError(f"Expected tile image shape (H, W, C), got {array.shape}.")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    else:
        array = array[:, :, :3]
    if array.dtype == np.uint8:
        return array
    stretched = np.zeros(array.shape, dtype=np.uint8)
    for band_index in range(3):
        band = np.asarray(array[:, :, band_index], dtype=np.float32)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, (2.0, 98.0))
        if high <= low:
            high = low + 1.0
        stretched[:, :, band_index] = np.clip(
            (band - low) * 255.0 / (high - low), 0.0, 255.0
        ).astype(np.uint8)
    return stretched


class TileReviewApplication:
    """Matplotlib event controller for navigation, selection, and status review."""

    def __init__(self, args, manifest_path, manifest, paths, plt, np):
        """Initialize session state and connect existing Matplotlib event hooks."""
        if args.selection_distance < 0:
            raise ValueError("--selection-distance must be non-negative.")
        if args.bbox_nearest_threshold < 0:
            raise ValueError("--bbox-nearest-threshold must be non-negative.")
        if args.add_bbox_size <= 0:
            raise ValueError("--add-bbox-size must be positive.")
        if args.negative_point_tolerance < 0:
            raise ValueError("--negative-point-tolerance must be non-negative.")
        if args.max_polygon_points < 3:
            raise ValueError("--max-polygon-points must be at least 3.")
        if args.min_polygon_area <= 0:
            raise ValueError("--min-polygon-area must be greater than zero.")
        if args.max_area_change_ratio <= 0 or args.min_area_change_ratio <= 0:
            raise ValueError("--min/max-area-change-ratio must be greater than zero.")
        if not str(args.prompt_source_name).strip():
            raise ValueError("--prompt-source-name must not be empty.")
        self.args = args
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.paths = paths
        self.plt = plt
        self.np = np
        self.index = first_review_index(
            manifest.tiles, paths.edited_results, args.start_tile
        )
        self.result = None
        self.auto_result = None
        self.from_edited = False
        self.selected_polygon_id = None
        self.mode = self.args.default_mode
        self.prompt_review_session = None
        self.refine_session = None
        self.prompt_session_cache = None
        self.modified = False
        self._closing = False
        self._image_artist = None
        self._polygon_artists = []
        self._mode_artist = None
        self._draw_started_at = None
        self._pressed_navigation_keys = set()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("draw_event", self.on_draw)
        default_key_handler_id = getattr(
            self.fig.canvas.manager, "key_press_handler_id", None
        )
        if default_key_handler_id is not None:
            self.fig.canvas.mpl_disconnect(default_key_handler_id)
            print("[review] Matplotlib default key handler disabled.")
        try:
            self.fig.canvas.manager.set_window_title("SAMPoly Tile Review")
        except AttributeError:
            pass
        self.load_current_tile()

    def refine_validation_config(self):
        """Build one session validation config from current command-line arguments."""
        return RefinePolygonValidationConfig(
            max_polygon_points=int(self.args.max_polygon_points),
            min_polygon_area=float(self.args.min_polygon_area),
            max_area_change_ratio=float(self.args.max_area_change_ratio),
            min_area_change_ratio=float(self.args.min_area_change_ratio),
            reject_invalid_refine=bool(self.args.reject_invalid_refine),
        )

    @property
    def tile(self):
        """Return the current manifest tile entry."""
        return self.manifest.tiles[self.index]

    def load_current_tile(self):
        """Load current image and edited/auto/empty result, then redraw the overlay."""
        started_at = time.perf_counter()
        self.auto_result = load_auto_result(self.tile, self.paths.auto_results)
        self.result, self.from_edited = load_review_result(
            self.tile,
            self.paths.auto_results,
            self.paths.edited_results,
        )
        if self.args.default_mode == "prompt":
            self.result.polygons = [
                polygon for polygon in self.result.polygons
                if polygon.source != "auto"
            ]
        image_path = resolve_tile_image(self.paths.workspace, self.tile.image_path)
        self.image_path = image_path
        with ImageTileReader(image_path, bands=(1, 2, 3)) as reader:
            image = reader.read_tile(0, 0, reader.width, reader.height)
        self.image = display_image(image, self.np)
        self.selected_polygon_id = None
        self.mode = self.args.default_mode
        self.prompt_review_session = None
        self.refine_session = None
        self.prompt_session_cache = None
        self.modified = False
        self.draw_overlay(reset_image=True)
        print(
            f"[review] tile load completed in {time.perf_counter() - started_at:.3f}s: "
            f"{self.tile.tile_id}"
        )

    def draw_overlay(self, reset_image=False):
        """Update polygon artists while retaining the current tile image when possible."""
        started_at = time.perf_counter()
        if reset_image or self._image_artist is None:
            self.ax.clear()
            self._image_artist = self.ax.imshow(self.image)
            image_height, image_width = self.image.shape[:2]
            self.ax.set_autoscale_on(False)
            self.ax.set_xlim(-0.5, image_width - 0.5)
            self.ax.set_ylim(image_height - 0.5, -0.5)
            self._polygon_artists = []
            self._mode_artist = None
            self.ax.axis("off")
        else:
            for artist in self._polygon_artists:
                artist.remove()
            self._polygon_artists = []
            if self._mode_artist is not None:
                self._mode_artist.remove()
                self._mode_artist = None

        visible_polygons = []
        if self.args.show_auto and self.args.default_mode == "prompt":
            for polygon in self.auto_result.polygons:
                points = self.np.asarray(polygon.points_tile, dtype=float)
                self._polygon_artists.extend(
                    self.ax.plot(
                        points[:, 0], points[:, 1],
                        color="royalblue", linewidth=1.0, linestyle="--", alpha=0.7, clip_on=True,
                    )
                )
        for polygon in self.result.polygons:
            display_polygon = (
                self.refine_session.current_polygon
                if self.refine_session is not None and polygon.id == self.refine_session.polygon_id
                else polygon
            )
            visible_polygons.append(polygon)
            points = self.np.asarray(display_polygon.points_tile, dtype=float)
            selected = polygon.id == self.selected_polygon_id
            if selected:
                color, linewidth, linestyle = "yellow", 3.0, "-"
            elif polygon.status == "deleted":
                color, linewidth, linestyle = "red", 1.2, "-"
            elif polygon.status == "hidden":
                color, linewidth, linestyle = "dimgray", 1.0, ":"
            else:
                color, linewidth, linestyle = "lime", 1.5, "-"
            artists = self.ax.plot(
                points[:, 0], points[:, 1],
                color=color, linewidth=linewidth, linestyle=linestyle, clip_on=True,
            )
            self._polygon_artists.extend(artists)
        if self.prompt_review_session is not None and self.prompt_review_session.current_polygon is not None:
            preview = self.np.asarray(
                self.prompt_review_session.current_polygon.points_tile,
                dtype=float,
            )
            self._polygon_artists.extend(
                self.ax.plot(
                    preview[:, 0], preview[:, 1],
                    color="cyan", linewidth=2.5, linestyle="-", clip_on=True,
                )
            )
        if self.prompt_review_session is not None and self.prompt_review_session.positive_points:
            clicks = self.np.asarray(self.prompt_review_session.positive_points, dtype=float)
            self._polygon_artists.append(self.ax.scatter(
                clicks[:, 0], clicks[:, 1], marker="+", s=90,
                linewidths=2.0, color="cyan", clip_on=True,
            ))
        if self.prompt_review_session is not None and self.prompt_review_session.negative_points:
            clicks = self.np.asarray(self.prompt_review_session.negative_points, dtype=float)
            self._polygon_artists.append(self.ax.scatter(
                clicks[:, 0], clicks[:, 1], marker="x", s=85,
                linewidths=2.0, color="red", clip_on=True,
            ))
        if self.refine_session is not None and self.refine_session.positive_points:
            clicks = self.np.asarray(self.refine_session.positive_points, dtype=float)
            self._polygon_artists.append(self.ax.scatter(
                clicks[:, 0], clicks[:, 1], marker="+", s=90,
                linewidths=2.0, color="cyan", clip_on=True,
            ))
        if self.refine_session is not None and self.refine_session.negative_points:
            clicks = self.np.asarray(self.refine_session.negative_points, dtype=float)
            self._polygon_artists.append(self.ax.scatter(
                clicks[:, 0], clicks[:, 1], marker="x", s=85,
                linewidths=2.0, color="red", clip_on=True,
            ))
        active = sum(
            polygon.status == "active" for polygon in visible_polygons
        )
        deleted = sum(
            polygon.status == "deleted" for polygon in visible_polygons
        )
        source_label = "edited_results" if self.from_edited else "auto/empty"
        modified_label = "  |  modified" if self.modified else ""
        self.ax.set_title(
            f"{self.tile.tile_id}  |  {self.index + 1}/{len(self.manifest.tiles)}  |  "
            f"active={active} deleted={deleted}  |  source={source_label}{modified_label}"
        )
        active_session = self.prompt_review_session or self.refine_session
        prompt_session_state = (
            active_session.state.value
            if active_session is not None else "idle"
        )
        prompt_bbox_source = (
            self.prompt_review_session.bbox_source
            if self.prompt_review_session is not None else
            ("refine_session_bbox" if self.refine_session is not None else "-")
        )
        prompt_positive_count = (
            len(active_session.positive_points)
            if active_session is not None else 0
        )
        prompt_negative_count = (
            len(active_session.negative_points)
            if active_session is not None else 0
        )
        self._mode_artist = self.ax.text(
            0.01,
            0.99,
            "\n".join(
                [
                    f"Mode: {self.mode.title()}",
                    f"Session: {prompt_session_state}",
                    f"BBox: {prompt_bbox_source}",
                    f"Points: +{prompt_positive_count} / -{prompt_negative_count}",
                    "PointRefine: "
                    f"{self.args.point_refine_mode if self.args.refine_by_points else 'off'}",
                    f"NegFeature: {'on' if self.args.negative_feature_refine else 'off'}",
                ]
            ),
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=11,
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
        self._draw_started_at = time.perf_counter()
        self.fig.canvas.draw_idle()
        print(
            f"[review] overlay update scheduled in "
            f"{time.perf_counter() - started_at:.3f}s: polygons={len(self.result.polygons)}"
        )

    def on_draw(self, _event):
        """Report the real canvas rendering duration after a scheduled overlay update."""
        if self._draw_started_at is None:
            self.restore_canvas_focus()
            return
        print(
            f"[review] canvas render completed in "
            f"{time.perf_counter() - self._draw_started_at:.3f}s"
        )
        self._draw_started_at = None
        self.restore_canvas_focus()

    def restore_canvas_focus(self):
        """Return keyboard focus to common Matplotlib GUI canvas backends."""
        canvas = self.fig.canvas
        set_focus = getattr(canvas, "setFocus", None)
        if callable(set_focus):
            set_focus()
            return
        grab_focus = getattr(canvas, "grab_focus", None)
        if callable(grab_focus):
            grab_focus()
            return
        get_tk_widget = getattr(canvas, "get_tk_widget", None)
        if callable(get_tk_widget):
            focus_set = getattr(get_tk_widget(), "focus_set", None)
            if callable(focus_set):
                focus_set()

    def save_current_tile(self):
        """Save current review state and mark subsequent redraws as edited."""
        started_at = time.perf_counter()
        path = save_review_result(self.result, self.paths.edited_results)
        self.from_edited = True
        self.modified = False
        print(
            f"[review] save completed in {time.perf_counter() - started_at:.3f}s: "
            f"{self.tile.tile_id} -> {path}"
        )
        return path

    def navigate(self, step):
        """Save and move one tile while stopping at manifest boundaries."""
        if self.prompt_review_session is not None:
            print("Accept with Enter or cancel with Esc/C before changing tiles.")
            return
        if self.refine_session is not None:
            print("Accept with Enter or cancel with Esc before changing tiles.")
            return
        started_at = time.perf_counter()
        print(f"[review] navigation requested: step={step}, tile={self.tile.tile_id}")
        self.save_current_tile()
        target = self.index + step
        if target < 0 or target >= len(self.manifest.tiles):
            print("Already at the first tile." if target < 0 else "Already at the last tile.")
            self.draw_overlay()
            return
        self.index = target
        self.load_current_tile()
        print(
            f"[review] navigation completed in {time.perf_counter() - started_at:.3f}s: "
            f"tile={self.tile.tile_id}"
        )

    def on_click(self, event):
        """Select a containing polygon or nearest boundary for a left-button click."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button not in (1, 3):
            return
        click = (float(event.xdata), float(event.ydata))
        if self.mode == "prompt":
            if self.prompt_review_session is None:
                if event.button == 1:
                    self.start_prompt_review(click)
                return
            self.update_prompt_review(click, is_positive=(event.button == 1))
            return
        if self.mode == "add":
            if event.button != 1:
                return
            self.add_polygon_at_click(click)
            return
        if self.mode == "refine":
            self.refine_polygon_at_click(click, is_positive=(event.button == 1))
            return
        if event.button != 1:
            return
        polygon = select_polygon_at_point(
            self.result,
            click,
            max_distance=self.args.selection_distance,
        )
        self.selected_polygon_id = polygon.id if polygon else None
        if polygon is None:
            print("No polygon selected.")
        else:
            print(
                f"Selected polygon: id={polygon.id}, status={polygon.status}, "
                f"score={polygon.score}, source={polygon.source}"
            )
        self.draw_overlay()

    def selected_polygon(self):
        """Return the selected canonical polygon without changing review state."""
        if self.selected_polygon_id is None:
            return None
        return next(
            (polygon for polygon in self.result.polygons
             if polygon.id == self.selected_polygon_id),
            None,
        )

    def start_prompt_review(self, click_point):
        """Start one Prompt-first building session from a tile click."""
        image_height, image_width = self.image.shape[:2]
        click = (
            min(max(int(round(click_point[0])), 0), image_width - 1),
            min(max(int(round(click_point[1])), 0), image_height - 1),
        )
        bbox_info = None
        if self.args.bbox_provider in ("auto", "auto_or_fixed"):
            bbox_info = select_bbox_for_click(
                polygons=self.auto_result.polygons,
                click=click,
                nearest_threshold=self.args.bbox_nearest_threshold,
                tile_width=image_width,
                tile_height=image_height,
            )
        if self.args.bbox_provider == "fixed":
            bbox = make_add_bbox(click, image_width, image_height, self.args.add_bbox_size)
            bbox_source = "fixed_fallback"
            selected_auto_polygon_id = None
            selected_auto_score = None
            selected_auto_distance = None
        elif bbox_info is not None:
            bbox = bbox_info["bbox"]
            bbox_source = bbox_info["source"]
            selected_auto_polygon_id = bbox_info["polygon_id"]
            selected_auto_score = bbox_info["score"]
            selected_auto_distance = bbox_info["distance"]
        elif self.args.bbox_provider == "auto":
            print("Prompt-first mode could not find a usable Auto bbox for this click.")
            return
        else:
            bbox = make_add_bbox(click, image_width, image_height, self.args.add_bbox_size)
            bbox_source = "fixed_fallback"
            selected_auto_polygon_id = None
            selected_auto_score = None
            selected_auto_distance = None

        self.prompt_review_session = PromptReviewSession.start(
            tile_id=self.tile.tile_id,
            bbox=bbox,
            bbox_source=bbox_source,
            selected_auto_polygon_id=selected_auto_polygon_id,
            selected_auto_score=selected_auto_score,
            selected_auto_distance=selected_auto_distance,
            validation_config=self.refine_validation_config(),
        )
        self.prompt_session_cache = PromptSessionCache()
        self.mode = "prompt"
        print(
            f"Prompt session started: bbox_source={bbox_source}, "
            f"auto_polygon={selected_auto_polygon_id}"
        )
        self.update_prompt_review(click, is_positive=True)

    def update_prompt_review(self, click_point, is_positive):
        """Add one positive or negative Prompt point and refresh the preview."""
        if self.prompt_review_session is None:
            raise RuntimeError("Prompt mode has no active session.")
        image_height, image_width = self.image.shape[:2]
        click = (
            min(max(int(round(click_point[0])), 0), image_width - 1),
            min(max(int(round(click_point[1])), 0), image_height - 1),
        )
        bbox = self.prompt_review_session.bbox
        if not is_positive and not self.prompt_review_session.can_accept_negative_point(
            click,
            tolerance=self.args.negative_point_tolerance,
            prefer_polygon=True,
            allow_outside_negative_points=self.args.allow_outside_negative_points,
        ):
            debug_info = dict(self.prompt_review_session.last_negative_point_debug)
            if self.args.debug_prompt_points:
                print("[debug_prompt_points] negative_point_validation =", {
                    **debug_info,
                    "image_shape": [int(image_height), int(image_width)],
                })
            print(
                "Negative point outside current preview polygon, ignored. "
                f"dist={debug_info.get('point_polygon_distance')}, "
                f"tolerance={debug_info.get('tolerance')}"
            )
            return
        if not is_positive and self.args.debug_prompt_points:
            print("[debug_prompt_points] negative_point_validation =", {
                **dict(self.prompt_review_session.last_negative_point_debug),
                "image_shape": [int(image_height), int(image_width)],
            })

        def prompt_callback(positive_points, negative_points, prompt_points=None, prompt_labels=None, click_order=None):
            if self.args.mock_prompt:
                return run_mock_prompt_inference(positive_points, negative_points, bbox)
            return run_prompt_inference_on_tile(
                image_path=self.image_path,
                positive_points=positive_points,
                negative_points=negative_points,
                bbox=bbox,
                config_path=Path(self.args.config),
                checkpoint_path=Path(self.args.checkpoint),
                device=self.args.device,
                session_cache=self.prompt_session_cache,
                prompt_points=prompt_points,
                prompt_labels=prompt_labels,
                debug_prompt_points=self.args.debug_prompt_points,
                refine_by_points=self.args.refine_by_points,
                point_refine_mode=self.args.point_refine_mode,
                negative_feature_refine=self.args.negative_feature_refine,
                negative_similarity_thr=self.args.negative_similarity_thr,
                negative_feature_patch_size=self.args.negative_feature_patch_size,
                negative_protect_radius=self.args.negative_protect_radius,
                negative_spatial_sigma=self.args.negative_spatial_sigma,
                debug_refine_dir=Path(self.args.debug_refine_dir) if self.args.debug_refine_dir else None,
                tile_id=self.tile.tile_id,
                session_id=self.prompt_review_session.session_id,
                click_order=click_order,
            )

        started_at = time.perf_counter()
        try:
            if not self.args.mock_prompt and (not self.args.config or not self.args.checkpoint):
                raise ValueError(
                    "Prompt mode requires --config and --checkpoint, or use --mock-prompt."
                )
            polygon = (
                self.prompt_review_session.add_positive_point(click, prompt_callback)
                if is_positive else
                self.prompt_review_session.add_negative_point(click, prompt_callback)
            )
        except Exception as exc:
            print(f"Prompt session failed: {type(exc).__name__}: {exc}")
            if not self.prompt_review_session.is_modified:
                self._close_prompt_review_session()
            return
        point_type = "positive" if is_positive else "negative"
        validation = dict(self.prompt_review_session.last_refine_validation)
        if validation and not validation.get("accepted", True):
            print(
                f"[review] prompt {point_type} click rejected: "
                f"reason={validation.get('reason')} "
                f"area={validation.get('candidate_area')} "
                f"ratio={validation.get('area_ratio')}"
            )
            self.draw_overlay()
            return
        print(
            f"[review] prompt {point_type} click completed in "
            f"{time.perf_counter() - started_at:.3f}s: "
            f"+{len(self.prompt_review_session.positive_points)} "
            f"-{len(self.prompt_review_session.negative_points)} "
            f"score={polygon.score}"
        )
        self.draw_overlay()

    def accept_prompt_review(self):
        """Commit the current Prompt-first preview as one confirmed saved polygon."""
        if self.prompt_review_session is None:
            return
        polygon_id = next_source_polygon_id(self.result, self.args.prompt_source_name)
        accepted = self.prompt_review_session.accept(
            polygon_id=polygon_id,
            source_name=self.args.prompt_source_name,
        )
        self.result.polygons.append(accepted)
        session_id = self.prompt_review_session.session_id
        self.selected_polygon_id = accepted.id
        self._close_prompt_review_session()
        self.modified = True
        self.save_current_tile()
        print(f"Prompt session accepted and saved: {session_id} -> {accepted.id}")
        self.draw_overlay()

    def cancel_prompt_review(self):
        """Discard one pending Prompt-first session without saving it."""
        if self.prompt_review_session is None:
            return
        session_id = self.prompt_review_session.session_id
        self.prompt_review_session.cancel()
        self._close_prompt_review_session()
        print(f"Prompt session cancelled: {session_id}")
        self.draw_overlay()

    def _close_prompt_review_session(self):
        """Release Prompt-first session state and cached Prompt resources."""
        if self.prompt_session_cache is not None:
            self.prompt_session_cache.clear()
        self.prompt_review_session = None
        self.prompt_session_cache = None
        self.mode = self.args.default_mode

    def start_refine(self):
        """Start an in-memory Refine session for the selected polygon."""
        polygon = self.selected_polygon()
        if polygon is None:
            print("Select a polygon before starting Refine mode.")
            return
        if not self.args.mock_prompt and (not self.args.config or not self.args.checkpoint):
            print("Refine mode requires --config and --checkpoint, or use --mock-prompt.")
            return
        self.refine_session = RefineSession.start(
            polygon,
            validation_config=self.refine_validation_config(),
        )
        self.prompt_session_cache = PromptSessionCache()
        self.mode = "refine"
        print(
            f"Refine mode enabled: polygon={polygon.id}. "
            "Left-click positive points; Enter accepts; Esc cancels."
        )
        self.draw_overlay()

    def refine_polygon_at_click(self, click_point, is_positive=True):
        """Add one positive or negative point and preview a new polygon."""
        if self.refine_session is None:
            raise RuntimeError("Refine mode has no active session.")
        image_height, image_width = self.image.shape[:2]
        click = (
            min(max(int(round(click_point[0])), 0), image_width - 1),
            min(max(int(round(click_point[1])), 0), image_height - 1),
        )
        original = self.refine_session.original_polygon
        bbox = original.bbox_tile or [
            min(point[0] for point in original.points_tile),
            min(point[1] for point in original.points_tile),
            max(point[0] for point in original.points_tile),
            max(point[1] for point in original.points_tile),
        ]
        if not is_positive and not self.refine_session.can_accept_negative_point(
            click,
            tolerance=self.args.negative_point_tolerance,
            prefer_polygon=True,
            allow_outside_negative_points=self.args.allow_outside_negative_points,
        ):
            debug_info = dict(self.refine_session.last_negative_point_debug)
            if self.args.debug_prompt_points:
                print("[debug_prompt_points] negative_point_validation =", {
                    **debug_info,
                    "image_shape": [int(image_height), int(image_width)],
                })
            print(
                "Negative point outside current preview polygon, ignored. "
                f"dist={debug_info.get('point_polygon_distance')}, "
                f"tolerance={debug_info.get('tolerance')}"
            )
            return
        if not is_positive and self.args.debug_prompt_points:
            print("[debug_prompt_points] negative_point_validation =", {
                **dict(self.refine_session.last_negative_point_debug),
                "image_shape": [int(image_height), int(image_width)],
            })

        def prompt_callback(positive_points, negative_points, prompt_points=None, prompt_labels=None, click_order=None):
            if self.args.mock_prompt:
                return run_mock_prompt_inference(positive_points, negative_points, bbox)
            return run_prompt_inference_on_tile(
                image_path=self.image_path,
                positive_points=positive_points,
                negative_points=negative_points,
                bbox=bbox,
                config_path=Path(self.args.config),
                checkpoint_path=Path(self.args.checkpoint),
                device=self.args.device,
                session_cache=self.prompt_session_cache,
                prompt_points=prompt_points,
                prompt_labels=prompt_labels,
                debug_prompt_points=self.args.debug_prompt_points,
                refine_by_points=self.args.refine_by_points,
                point_refine_mode=self.args.point_refine_mode,
                negative_feature_refine=self.args.negative_feature_refine,
                negative_similarity_thr=self.args.negative_similarity_thr,
                negative_feature_patch_size=self.args.negative_feature_patch_size,
                negative_protect_radius=self.args.negative_protect_radius,
                negative_spatial_sigma=self.args.negative_spatial_sigma,
                debug_refine_dir=Path(self.args.debug_refine_dir) if self.args.debug_refine_dir else None,
                tile_id=self.tile.tile_id,
                session_id=self.refine_session.session_id,
                click_order=click_order,
            )

        started_at = time.perf_counter()
        try:
            polygon = (
                self.refine_session.add_positive_point(click, prompt_callback)
                if is_positive else
                self.refine_session.add_negative_point(click, prompt_callback)
            )
        except Exception as exc:
            print(f"Refine Prompt failed: {type(exc).__name__}: {exc}")
            return
        validation = dict(self.refine_session.last_refine_validation)
        if validation and not validation.get("accepted", True):
            print(
                f"[review] refine Prompt rejected: "
                f"reason={validation.get('reason')} "
                f"area={validation.get('candidate_area')} "
                f"ratio={validation.get('area_ratio')}"
            )
            self.draw_overlay()
            return
        print(
            f"[review] refine Prompt completed in {time.perf_counter() - started_at:.3f}s: "
            f"+{len(self.refine_session.positive_points)} "
            f"-{len(self.refine_session.negative_points)} score={polygon.score}"
        )
        self.draw_overlay()

    def accept_refine(self):
        """Commit the current preview polygon and immediately save edited_results."""
        if self.refine_session is None:
            return
        accepted = self.refine_session.accept()
        for index, polygon in enumerate(self.result.polygons):
            if polygon.id == accepted.id:
                self.result.polygons[index] = accepted
                break
        else:
            raise KeyError(f"Refined polygon disappeared from TileResult: {accepted.id}")
        session_id = self.refine_session.session_id
        self._close_refine_session()
        self.modified = True
        self.save_current_tile()
        print(f"Refine session accepted and saved: {session_id}")
        self.draw_overlay()

    def cancel_refine(self):
        """Discard all session previews and leave the TileResult untouched."""
        if self.refine_session is None:
            return
        session_id = self.refine_session.session_id
        self.refine_session.cancel()
        self._close_refine_session()
        print(f"Refine session cancelled: {session_id}")
        self.draw_overlay()

    def _close_refine_session(self):
        """Release Refine-only state and cached image references."""
        if self.prompt_session_cache is not None:
            self.prompt_session_cache.clear()
        self.refine_session = None
        self.prompt_session_cache = None
        self.mode = self.args.default_mode

    def add_polygon_at_click(self, click_point):
        """Run mock/real Prompt inference and append one missed building."""
        started_at = time.perf_counter()
        image_height, image_width = self.image.shape[:2]
        click = (
            min(max(int(round(click_point[0])), 0), image_width - 1),
            min(max(int(round(click_point[1])), 0), image_height - 1),
        )
        try:
            bbox = make_add_bbox(
                click,
                image_width,
                image_height,
                self.args.add_bbox_size,
            )
            if self.args.mock_prompt:
                prediction = run_mock_prompt_inference([click], [], bbox)
            else:
                if not self.args.config or not self.args.checkpoint:
                    print(
                        "Add mode requires --config and --checkpoint, or use --mock-prompt."
                    )
                    return
                prediction = run_prompt_inference_on_tile(
                    image_path=self.image_path,
                    positive_points=[click],
                    negative_points=[],
                    bbox=bbox,
                    config_path=Path(self.args.config),
                    checkpoint_path=Path(self.args.checkpoint),
                    device=self.args.device,
                )
            print(
                f"[review] prompt inference completed in "
                f"{time.perf_counter() - started_at:.3f}s"
            )
            if prediction is None:
                raise ValueError("Prompt inference returned no polygon.")
            polygon = add_click_polygon(self.result, prediction, click)
        except Exception as exc:
            print(f"Add polygon failed: {type(exc).__name__}: {exc}")
            return

        self.selected_polygon_id = polygon.id
        self.modified = True
        self.mode = self.args.default_mode
        print(f"Added polygon from click: id={polygon.id}, score={polygon.score}")
        self.draw_overlay()

    def change_selected_status(self, status):
        """Apply a soft-delete or restore operation to the current selection."""
        if self.selected_polygon_id is None:
            print("No polygon is selected.")
            return
        polygon = set_polygon_status(self.result, self.selected_polygon_id, status)
        self.modified = True
        print(f"Polygon {polygon.id} status changed to {polygon.status}.")
        self.draw_overlay()

    def on_key(self, event):
        """Dispatch documented keyboard shortcuts without invoking model operations."""
        key = (event.key or "").lower()
        navigation_step = navigation_step_for_key(key)
        native_repeat_supported = callable(
            getattr(getattr(event, "guiEvent", None), "isAutoRepeat", None)
        )
        if self.mode == "refine":
            if key in ("enter", "return"):
                self.accept_refine()
            elif key == "escape":
                self.cancel_refine()
            elif key == "s":
                print("Refine preview is memory-only; press Enter to accept before saving it.")
            elif key == "h":
                print(HELP_TEXT)
            elif key == "q":
                self.cancel_refine()
                self.save_current_tile()
                self._closing = True
                self.plt.close(self.fig)
            else:
                print("Refine mode is running; use left/right click, Enter, Esc, or H.")
            return
        if self.prompt_review_session is not None:
            if key in ("enter", "return"):
                self.accept_prompt_review()
            elif key in ("escape", "c"):
                self.cancel_prompt_review()
            elif key == "s":
                print("Prompt preview is memory-only; press Enter to accept before saving it.")
            elif key == "h":
                print(HELP_TEXT)
            elif key == "q":
                self.cancel_prompt_review()
                self.save_current_tile()
                self._closing = True
                self.plt.close(self.fig)
            else:
                print("Prompt session is running; use left/right click, Enter, C/Esc, or H.")
            return
        if navigation_step is not None:
            if is_auto_repeat_key_event(event):
                return
            if not native_repeat_supported and key in self._pressed_navigation_keys:
                return
            self._pressed_navigation_keys.add(key)
            print(f"[review] navigation key received: {key!r}")
            self.navigate(navigation_step)
        elif key == "d":
            print("[review] command key received: 'd'")
            self.change_selected_status("deleted")
        elif key == "u":
            print("[review] command key received: 'u'")
            self.change_selected_status("active")
        elif key == "a":
            print("[review] command key received: 'a'")
            self.mode = "add"
            self.selected_polygon_id = None
            print("Add mode enabled. Left-click one missed building; Esc cancels.")
            self.draw_overlay()
        elif key == "r":
            print("[review] command key received: 'r'")
            self.start_refine()
        elif key == "s":
            self.save_current_tile()
            self.draw_overlay()
        elif key in ("escape", "c") and self.mode == "add":
            self.mode = self.args.default_mode
            print("Add mode cancelled.")
            self.draw_overlay()
        elif key in ("q", "escape"):
            self.save_current_tile()
            self._closing = True
            self.plt.close(self.fig)
        elif key == "h":
            print(HELP_TEXT)

    def on_key_release(self, event):
        """Release the non-Qt repeat guard for one navigation key."""
        key = (event.key or "").lower()
        self._pressed_navigation_keys.discard(key)

    def on_close(self, _event):
        """Save when the window manager closes the figure directly."""
        if not self._closing and self.result is not None:
            self._closing = True
            if self.prompt_review_session is not None:
                self.prompt_review_session.cancel()
                self._close_prompt_review_session()
            if self.refine_session is not None:
                self.refine_session.cancel()
                self._close_refine_session()
            self.save_current_tile()

    def show(self):
        """Print controls and enter the Matplotlib GUI event loop."""
        print(HELP_TEXT)
        self.plt.show()


def main() -> int:
    """Resolve the review session and launch the GUI with explicit errors."""
    try:
        args = parse_args()
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        manifest = TileManifest.load(manifest_path)
        paths = resolve_review_paths(
            manifest_path,
            Path(args.auto_results) if args.auto_results else None,
            Path(args.edited_results) if args.edited_results else None,
        )
        import matplotlib.pyplot as plt
        import numpy as np

        TileReviewApplication(args, manifest_path, manifest, paths, plt, np).show()
    except Exception as exc:
        print(f"Tile review failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
