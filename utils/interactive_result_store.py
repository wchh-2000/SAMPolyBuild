import json
import math
import os


VALID_SOURCES = {"auto", "prompt", "edited"}
VALID_STATUSES = {"active", "deleted"}


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height].")
    values = [float(value) for value in bbox]
    if not all(_finite_number(value) for value in values):
        raise ValueError("bbox contains a non-finite coordinate.")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("bbox width and height must be positive.")
    return values


def normalize_polygon(segmentation):
    """Return one closed [[x, y], ...] ring from COCO or point-pair input."""
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        raise ValueError("polygon is empty or missing.")
    candidate = segmentation
    if isinstance(candidate[0], (list, tuple)):
        if candidate[0] and isinstance(candidate[0][0], (list, tuple)):
            candidate = max(candidate, key=len)
        elif candidate[0] and _finite_number(candidate[0][0]) and len(candidate[0]) != 2:
            candidate = max(candidate, key=len)
    if candidate and _finite_number(candidate[0]):
        if len(candidate) < 6 or len(candidate) % 2:
            raise ValueError("flattened polygon must contain at least three x/y pairs.")
        points = [[float(candidate[index]), float(candidate[index + 1])]
                  for index in range(0, len(candidate), 2)]
    else:
        points = []
        for point in candidate:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("polygon points must be [x, y] pairs.")
            points.append([float(point[0]), float(point[1])])
    if len(points) < 3 or not all(_finite_number(value) for point in points for value in point):
        raise ValueError("polygon must contain at least three finite points.")
    if points[0] != points[-1]:
        points.append(points[0][:])
    if len(points) < 4:
        raise ValueError("closed polygon must contain at least four coordinates.")
    return points


def polygon_bbox(points):
    points = normalize_polygon(points)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _point_in_polygon(point, polygon):
    x, y = point
    inside = False
    for first, second in zip(polygon, polygon[1:]):
        x1, y1 = first
        x2, y2 = second
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= crossing_x:
                inside = not inside
    return inside


def _bbox_contains(point, bbox, tolerance=0.0):
    x, y, width, height = bbox
    return x - tolerance <= point[0] <= x + width + tolerance and \
        y - tolerance <= point[1] <= y + height + tolerance


class InteractiveResultStore:
    def __init__(self, image_name, tile_offset=(0, 0), output_path=None):
        self.image_name = image_name
        self.tile_offset = [float(tile_offset[0]), float(tile_offset[1])]
        self.output_path = output_path
        self.instances = {}

    def local_to_global_point(self, point):
        return [float(point[0]) + self.tile_offset[0], float(point[1]) + self.tile_offset[1]]

    def global_to_local_point(self, point):
        return [float(point[0]) - self.tile_offset[0], float(point[1]) - self.tile_offset[1]]

    def local_to_global_bbox(self, bbox_xyxy):
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        return [x1 + self.tile_offset[0], y1 + self.tile_offset[1], x2 - x1, y2 - y1]

    def global_to_local_bbox_xyxy(self, bbox):
        x, y, width, height = normalize_bbox(bbox)
        return [x - self.tile_offset[0], y - self.tile_offset[1],
                x + width - self.tile_offset[0], y + height - self.tile_offset[1]]

    def global_to_local_polygon(self, polygon):
        return [self.global_to_local_point(point) for point in normalize_polygon(polygon)]

    def local_to_global_polygon(self, polygon):
        return [self.local_to_global_point(point) for point in normalize_polygon(polygon)]

    def load_initial_results(self, path, image_id=None, viewport_size=None, min_score=0.0):
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict) and "instances" in payload:
            records = list(payload["instances"].values())
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError("initial_results must be a prediction list or an object with instances.")

        loaded = 0
        warnings = []
        for index, record in enumerate(records):
            try:
                if not isinstance(record, dict):
                    raise ValueError("record must be an object.")
                if image_id is not None and record.get("image_id") is not None \
                        and str(record.get("image_id")) != str(image_id):
                    continue
                score_value = record.get("score_cls", record.get("score", 0.0))
                score = 0.0 if score_value is None else float(score_value)
                if score < min_score:
                    continue
                bbox = record.get("bbox", record.get("selected_bbox"))
                if "selected_bbox" in record and "bbox" not in record:
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                bbox = normalize_bbox(bbox)
                polygon_value = record.get("polygon", record.get("latest_polygon",
                                           record.get("segmentation")))
                polygon = normalize_polygon(polygon_value)
                instance_id = str(record.get("instance_id") or f"auto_{index:04d}")
                while instance_id in self.instances:
                    instance_id = f"{instance_id}_{index}"
                source = record.get("source", "auto")
                status = record.get("status", "active")
                prompt_points = record.get("prompt_points", record.get("click_points", []))
                prompt_labels = record.get("prompt_labels", [])
                if prompt_points and isinstance(prompt_points[0], dict):
                    prompt_labels = [int(point.get("label", 1)) for point in prompt_points]
                    prompt_points = [[float(point["x"]), float(point["y"])] for point in prompt_points]
                self.instances[instance_id] = {
                    "instance_id": instance_id,
                    "source": source if source in VALID_SOURCES else "auto",
                    "status": status if status in VALID_STATUSES else "active",
                    "bbox": bbox,
                    "polygon": polygon,
                    "score": score,
                    "tile_id": record.get("tile_id"),
                    "prompt_points": prompt_points,
                    "prompt_labels": prompt_labels,
                    "version": max(1, int(record.get("version", 1))),
                }
                loaded += 1
            except (TypeError, ValueError) as exc:
                warnings.append(f"initial result {index} skipped: {exc}")
        return loaded, warnings

    def _intersects_viewport(self, bbox, viewport_size):
        x1, y1, x2, y2 = self.global_to_local_bbox_xyxy(bbox)
        width, height = viewport_size
        return x2 >= 0 and y2 >= 0 and x1 <= width and y1 <= height

    def visible_instances(self, viewport_size):
        return [instance for instance in self.instances.values()
                if self._intersects_viewport(instance["bbox"], viewport_size)]

    def hit_test_local(self, point, tolerance=3.0):
        global_point = self.local_to_global_point(point)
        polygon_hits = []
        bbox_hits = []
        for instance in self.instances.values():
            if instance["status"] != "active":
                continue
            area = instance["bbox"][2] * instance["bbox"][3]
            if _point_in_polygon(global_point, instance["polygon"]):
                polygon_hits.append((area, instance["instance_id"]))
            elif _bbox_contains(global_point, instance["bbox"], tolerance):
                bbox_hits.append((area, instance["instance_id"]))
        hits = polygon_hits or bbox_hits
        return min(hits)[1] if hits else None

    def create_prompt(self, bbox_xyxy, polygon, prompt_points, prompt_labels, score=None):
        sequence = 1
        while f"prompt_{sequence:04d}" in self.instances:
            sequence += 1
        instance_id = f"prompt_{sequence:04d}"
        self.instances[instance_id] = {
            "instance_id": instance_id,
            "source": "prompt",
            "status": "active",
            "bbox": self.local_to_global_bbox(bbox_xyxy),
            "polygon": self.local_to_global_polygon(polygon),
            "score": None if score is None else float(score),
            "tile_id": None,
            "prompt_points": [self.local_to_global_point(point) for point in prompt_points],
            "prompt_labels": [int(label) for label in prompt_labels],
            "version": 1,
        }
        return instance_id

    def update_from_local(self, instance_id, bbox_xyxy, polygon, prompt_points, prompt_labels, score=None):
        if instance_id not in self.instances:
            raise KeyError(f"Unknown instance_id: {instance_id}")
        instance = self.instances[instance_id]
        if instance["status"] != "active":
            raise ValueError(f"Cannot edit deleted instance: {instance_id}")
        instance.update({
            "source": "edited",
            "bbox": self.local_to_global_bbox(bbox_xyxy),
            "polygon": self.local_to_global_polygon(polygon),
            "prompt_points": [self.local_to_global_point(point) for point in prompt_points],
            "prompt_labels": [int(label) for label in prompt_labels],
            "version": int(instance["version"]) + 1,
        })
        if score is not None:
            instance["score"] = float(score)
        return instance

    def mark_deleted(self, instance_id):
        if instance_id not in self.instances:
            raise KeyError(f"Unknown instance_id: {instance_id}")
        instance = self.instances[instance_id]
        if instance["status"] != "deleted":
            instance["status"] = "deleted"
            instance["version"] = int(instance["version"]) + 1
        return instance

    def save(self, path=None):
        destination = path or self.output_path
        if not destination:
            raise ValueError("An output path is required.")
        payload = {
            "schema_version": 1,
            "image_name": self.image_name,
            "coordinate_space": "global",
            "tile_offset": self.tile_offset,
            "instances": self.instances,
        }
        os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
        temporary = destination + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
        return destination
