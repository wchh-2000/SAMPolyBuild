"""Lightweight Matplotlib viewer for reviewing canonical per-tile polygons."""

import argparse
import sys
from pathlib import Path

from utils.image_tile_reader import ImageTileReader
from workflow.manifest import TileManifest
from workflow.tile_review import (
    first_review_index,
    load_review_result,
    polygon_status_counts,
    resolve_review_paths,
    resolve_tile_image,
    save_review_result,
    select_polygon_at_point,
    set_polygon_status,
)


HELP_TEXT = """Tile review shortcuts:
  N / Right Arrow  save and open next tile
  P / Left Arrow   save and open previous tile
  D                mark selected polygon deleted
  U                restore selected polygon to active
  S                save current tile
  Q / Esc          save and quit
  H                print this help
  Left click       select polygon
"""


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
        self.from_edited = False
        self.selected_polygon_id = None
        self._closing = False
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        try:
            self.fig.canvas.manager.set_window_title("SAMPoly Tile Review")
        except AttributeError:
            pass
        self.load_current_tile()

    @property
    def tile(self):
        """Return the current manifest tile entry."""
        return self.manifest.tiles[self.index]

    def load_current_tile(self):
        """Load current image and edited/auto/empty result, then redraw the overlay."""
        self.result, self.from_edited = load_review_result(
            self.tile,
            self.paths.auto_results,
            self.paths.edited_results,
        )
        image_path = resolve_tile_image(self.paths.workspace, self.tile.image_path)
        with ImageTileReader(image_path, bands=(1, 2, 3)) as reader:
            image = reader.read_tile(0, 0, reader.width, reader.height)
        self.image = display_image(image, self.np)
        self.selected_polygon_id = None
        self.draw_overlay()

    def draw_overlay(self):
        """Draw image, active/deleted polygons, selection highlight, and status title."""
        self.ax.clear()
        self.ax.imshow(self.image)
        for polygon in self.result.polygons:
            points = self.np.asarray(polygon.points_tile, dtype=float)
            selected = polygon.id == self.selected_polygon_id
            if selected:
                color, linewidth, linestyle = "yellow", 3.0, "-"
            elif polygon.status == "deleted":
                color, linewidth, linestyle = "gray", 1.2, "--"
            elif polygon.status == "hidden":
                color, linewidth, linestyle = "dimgray", 1.0, ":"
            else:
                color, linewidth, linestyle = "lime", 1.5, "-"
            self.ax.plot(
                points[:, 0], points[:, 1],
                color=color, linewidth=linewidth, linestyle=linestyle,
            )
        active, deleted = polygon_status_counts(self.result)
        source_label = "edited_results" if self.from_edited else "auto/empty"
        self.ax.set_title(
            f"{self.tile.tile_id}  |  {self.index + 1}/{len(self.manifest.tiles)}  |  "
            f"active={active} deleted={deleted}  |  source={source_label}"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def save_current_tile(self):
        """Save current review state and mark subsequent redraws as edited."""
        path = save_review_result(self.result, self.paths.edited_results)
        self.from_edited = True
        print(f"Saved {self.tile.tile_id}: {path}")
        return path

    def navigate(self, step):
        """Save and move one tile while stopping at manifest boundaries."""
        self.save_current_tile()
        target = self.index + step
        if target < 0 or target >= len(self.manifest.tiles):
            print("Already at the first tile." if target < 0 else "Already at the last tile.")
            self.draw_overlay()
            return
        self.index = target
        self.load_current_tile()

    def on_click(self, event):
        """Select a containing polygon or nearest boundary for a left-button click."""
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        polygon = select_polygon_at_point(
            self.result,
            (float(event.xdata), float(event.ydata)),
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

    def change_selected_status(self, status):
        """Apply a soft-delete or restore operation to the current selection."""
        if self.selected_polygon_id is None:
            print("No polygon is selected.")
            return
        polygon = set_polygon_status(self.result, self.selected_polygon_id, status)
        print(f"Polygon {polygon.id} status changed to {polygon.status}.")
        self.draw_overlay()

    def on_key(self, event):
        """Dispatch documented keyboard shortcuts without invoking model operations."""
        key = (event.key or "").lower()
        if key in ("n", "right"):
            self.navigate(1)
        elif key in ("p", "left"):
            self.navigate(-1)
        elif key == "d":
            self.change_selected_status("deleted")
        elif key == "u":
            self.change_selected_status("active")
        elif key == "s":
            self.save_current_tile()
            self.draw_overlay()
        elif key in ("q", "escape"):
            self.save_current_tile()
            self._closing = True
            self.plt.close(self.fig)
        elif key == "h":
            print(HELP_TEXT)

    def on_close(self, _event):
        """Save when the window manager closes the figure directly."""
        if not self._closing and self.result is not None:
            self._closing = True
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
