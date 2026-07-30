import argparse
import json
import os
import sys
from dataclasses import asdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.auto_bbox import AutoBBoxConfig, generate_adaptive_bbox


def parse_args():
    default_cfg = AutoBBoxConfig()
    parser = argparse.ArgumentParser(description="Interactively visualize adaptive bbox generation.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out", default="outputs/auto_bbox_visual", help="Output directory.")
    parser.add_argument("--debug", action="store_true", help="Keep and save debug mask when available.")

    for name, value in asdict(default_cfg).items():
        parser.add_argument(f"--{name.replace('_', '-')}", default=value, type=type(value))
    return parser.parse_args()


def build_config(args):
    keys = asdict(AutoBBoxConfig()).keys()
    return AutoBBoxConfig(**{key: getattr(args, key) for key in keys})


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def to_rgb_for_display(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def json_ready_info(info, out_dir):
    result = dict(info)
    debug_mask = result.get("debug_mask")
    if isinstance(debug_mask, np.ndarray):
        mask_path = os.path.join(out_dir, "mask.png")
        cv2.imwrite(mask_path, debug_mask)
        result["debug_mask"] = {
            "path": "mask.png",
            "shape": list(debug_mask.shape),
            "dtype": str(debug_mask.dtype),
            "nonzero": int(np.count_nonzero(debug_mask)),
        }
    return result


def print_summary(bbox, info):
    print("bbox:", bbox)
    print("method:", info.get("method"))
    print("fallback_used:", info.get("fallback_used"))
    print("message:", info.get("message"))


class AutoBBoxViewer:
    def __init__(self, image, display_image, config, out_dir, debug):
        self.image = image
        self.display_image = display_image
        self.config = config
        self.out_dir = out_dir
        self.debug = debug
        self.last_bbox = None
        self.last_info = None
        self.last_click = None
        self.artists = []

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.display_image)
        self.ax.axis("off")
        self.ax.set_title("Left click: generate bbox | c: clear | s: save | q/Esc: quit")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def clear_artists(self):
        for artist in self.artists:
            artist.remove()
        self.artists = []
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        point = (int(event.xdata), int(event.ydata))
        bbox, info = generate_adaptive_bbox(self.image, point, config=self.config, debug=self.debug)
        self.last_click = point
        self.last_bbox = bbox
        self.last_info = info
        print_summary(bbox, info)
        self.draw_result(point, bbox, info)

    def draw_result(self, point, bbox, info):
        self.clear_artists()
        roi = info.get("roi")
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            self.artists.append(
                self.ax.add_patch(
                    Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, edgecolor="cyan", linewidth=1.5)
                )
            )
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_color = "lime" if not info.get("fallback_used") else "orange"
        self.artists.append(
            self.ax.add_patch(
                Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=bbox_color, linewidth=2)
            )
        )
        self.artists.extend(self.ax.plot(point[0], point[1], marker="+", markersize=14, color="red"))
        status = f"{info.get('method')} | fallback={info.get('fallback_used')} | {info.get('message')}"
        self.ax.set_title(status)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == "c":
            self.last_bbox = None
            self.last_info = None
            self.last_click = None
            self.ax.set_title("Left click: generate bbox | c: clear | s: save | q/Esc: quit")
            self.clear_artists()
        elif event.key == "s":
            self.save_current()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def save_current(self):
        if self.last_bbox is None or self.last_info is None:
            print("No bbox to save. Click on the image first.")
            return
        os.makedirs(self.out_dir, exist_ok=True)
        self.fig.savefig(os.path.join(self.out_dir, "bbox_overlay.png"), bbox_inches="tight", pad_inches=0)
        info = json_ready_info(self.last_info, self.out_dir)
        info["saved_click_point"] = list(self.last_click)
        with open(os.path.join(self.out_dir, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"Saved auto bbox debug outputs to: {self.out_dir}")

    def show(self):
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()


def main():
    args = parse_args()
    try:
        image = load_image(args.image)
        display_image = to_rgb_for_display(image)
    except Exception as exc:
        print(f"Unable to start auto bbox viewer: {exc}")
        return 2
    viewer = AutoBBoxViewer(image, display_image, build_config(args), args.out, args.debug)
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
