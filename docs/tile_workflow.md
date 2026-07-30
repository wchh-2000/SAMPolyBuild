# Tile Workflow Foundation

## Purpose

The tile workflow separates large-image engineering concerns from SAMPoly model
inference. This foundation defines versioned JSON contracts, paths, and pixel
coordinate translation only. It does not import or alter Auto, Prompt, SAMPoly,
or other model code.

The planned flow is:

```text
source image
  -> manifest and tile images
  -> one auto TileResult per tile
  -> edited TileResult
  -> final results
```

Workflow data is organized under:

```text
data/
  manifests/
  auto_results/
  edited_results/
  final_results/
```

`TileManager` resolves these locations. Paths are represented internally with
`pathlib.Path`; JSON stores relative paths as POSIX strings for portability.

## Manifest schema

Every manifest uses `schema_version: 1`. The `metadata` objects reserve space
for geospatial transforms, validity statistics, filtering summaries, and other
future attributes without changing the stable core schema.

```json
{
  "schema_version": 1,
  "image_name": "source.tif",
  "image_width": 2048,
  "image_height": 1536,
  "tile_size": 1024,
  "overlap": 256,
  "tiles": [
    {
      "tile_id": "tile_0001",
      "image_path": "tiles/tile_0001.png",
      "x_offset": 0,
      "y_offset": 0,
      "width": 1024,
      "height": 1024,
      "is_edge_tile": false,
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

`TileManifest.load()` and `TileManifest.save()` provide validated, atomic JSON
persistence. Tile identifiers must be unique, dimensions must be positive, and
`0 <= overlap < tile_size`. Every retained tile must remain inside the declared
source-image pixel bounds.

## Polygon schema

```json
{
  "id": "auto_0001",
  "tile_id": "tile_0001",
  "status": "active",
  "source": "auto",
  "score": 0.91,
  "points_tile": [[0, 0], [10, 0], [10, 10], [0, 0]],
  "points_global": [],
  "click_history": [],
  "metadata": {}
}
```

Allowed lifecycle values are:

- `status`: `active`, `deleted`, `hidden`
- `source`: `auto`, `click`, `refine`

`points_tile` is required. `points_global`, `click_history`, and `metadata` are
reserved for later workflow stages. Polygon rings require at least three finite
vertices and are closed automatically during validation.

## TileResult schema

One tile corresponds to one JSON file:

```json
{
  "schema_version": 1,
  "tile_id": "tile_0001",
  "offset": [0, 0],
  "polygons": [],
  "metadata": {}
}
```

Use `TileResult.create_empty()` before a stage has produced polygons, and use
`TileResult.load()` and `TileResult.save()` for validated persistence. Polygon
ids must be unique within a tile and each polygon must reference the containing
TileResult's `tile_id`.

## Coordinate conversion

The first version supports pixel translation only:

```text
global_x = tile_x + x_offset
global_y = tile_y + y_offset

tile_x = global_x - x_offset
tile_y = global_y - y_offset
```

`tile_to_global()` and `global_to_tile()` each accept one `[x, y]` point and an
`[x_offset, y_offset]` pair. Rotation, scale, affine transforms, and CRS
conversion are deliberately outside this foundation.

## Later development stages

1. `tools/tile_large_image.py` emits this `TileManifest` at
   `output/manifests/manifest.json`. It also retains `output/tiles_manifest.json`
   for compatibility with existing inference and remapping tools.
2. Add adapters that convert existing Auto output into one `TileResult` per tile;
   do not modify Auto inference.
3. Store click and refinement output in `edited_results/`.
4. Populate `points_global` through the coordinate helpers.
5. Add final collection, validation, and GIS export under `final_results/`.

## Large-image tiling command

The CLI accepts both hyphenated workflow arguments and the original underscore
spellings. GeoTIFF remains the default output because it preserves geospatial
metadata; PNG and JPEG are optional output formats.

```bash
python tile_large_image.py \
  --input path/to/large_image.tif \
  --output-dir path/to/output \
  --tile-size 1024 \
  --overlap 256 \
  --tile-format png
```

The command writes row-major, continuously numbered tiles under `output/tiles/`,
the canonical manifest under `output/manifests/`, and the legacy compatibility
manifest at `output/tiles_manifest.json`.

## Per-tile Auto inference

The stage-2 command reuses `infer_auto.run_auto_inference()` and converts its
existing COCO polygon list into one canonical `TileResult` per tile:

```bash
python infer_tiles_auto.py \
  --manifest path/to/output/manifests/manifest.json \
  --output-dir path/to/output/auto_results \
  --config configs/auto_whumix.py \
  --checkpoint auto_whumix.pth \
  --device cuda:0 \
  --score-threshold 0.5
```

Existing tile JSON files are skipped unless `--overwrite` is supplied. Tiles
with no detections receive a valid empty result. The normal path performs one
batch inference for all pending tiles; if that batch fails, the workflow retries
tiles individually and records concrete errors without stopping the remaining
tiles.

## Tile review

The stage-3 Matplotlib viewer loads edited results first, falls back to Auto
results, and creates an empty TileResult when neither exists:

```bash
python review_tiles.py \
  --manifest path/to/output/manifests/manifest.json \
  --auto-results path/to/output/auto_results \
  --edited-results path/to/output/edited_results
```

Use `N`/Right and `P`/Left to save and navigate, click a polygon to select it,
`D` to mark it deleted, `U` to restore it, `S` to save, `H` for help, and
`Q`/Escape to save and quit. Press `A` to enter Add mode and click one missed
building; the new click polygon is written only to `edited_results`. Escape
cancels Add mode without closing the viewer. `--start-tile tile_0030` overrides the default
resume position, which is the first tile without an edited JSON file.

## Refine session foundation

Stage 6 adds an in-memory Refine session around the existing Prompt pipeline.
This keeps iterative Prompt clicks isolated from `edited_results` until the
user explicitly accepts the preview:

```text
Idle
  -> R starts Refine on the selected polygon
  -> left click appends one positive point
  -> Prompt re-runs with all accumulated session points
  -> current polygon preview updates in memory
  -> Enter accepts and writes the refined polygon
  -> Esc cancels and restores the original polygon
```

The session state lives in `workflow/refine_session.py` and stores:

- `original_polygon`
- `current_polygon`
- `positive_points`
- `negative_points` (reserved for the next stage)
- `history`
- `state`

`review_tiles.py` shows the active Refine session with `Mode: Refine`,
highlights the selected polygon preview, and overlays positive clicks in cyan.
Refine uses `workflow.prompt_infer_adapter.PromptSessionCache` so one tile image
and its prepared crop embedding can be reused across repeated Prompt updates.

Example command:

```bash
python review_tiles.py \
  --manifest path/to/output/manifests/manifest.json \
  --auto-results path/to/output/auto_results \
  --edited-results path/to/output/edited_results \
  --config configs/prompt_instance_spacenet.json \
  --checkpoint prompt_interactive.pth \
  --device cuda:0
```

For a model-free smoke check, replace the Prompt arguments with `--mock-prompt`.

## Prompt-first review mode

Stage 7 keeps `auto_results/` as a hidden bbox provider and makes Prompt the
default review entry point. The reviewer now starts in `prompt` mode unless
`--default-mode select` is specified.

```text
left click building
  -> select Auto bbox candidate when available
  -> fall back to fixed bbox when allowed
  -> run Prompt and show preview polygon
  -> left click adds positive points
  -> right click adds negative points only when they fall inside the current preview polygon/mask by default
  -> Enter saves the final prompt polygon
  -> C / Esc cancels the running prompt session
```

The default viewer behavior is:

- confirmed `prompt`, `refine`, `click`, and `manual` polygons stay visible
- `auto` polygons are hidden by default
- `auto` polygons can still provide Prompt bboxes in the background
- `--show-auto` draws Auto polygons for debugging

Example command:

```bash
python review_tiles.py \
  --manifest path/to/output/manifests/manifest.json \
  --auto-results path/to/output/auto_results \
  --edited-results path/to/output/edited_results \
  --config configs/prompt_instance_spacenet.json \
  --checkpoint prompt_interactive.pth \
  --device cuda:0
```

Legacy selection-first review is still available:

```bash
python review_tiles.py \
  --manifest path/to/output/manifests/manifest.json \
  --auto-results path/to/output/auto_results \
  --edited-results path/to/output/edited_results \
  --default-mode select
```

## Negative-point iterative refine

Stage 8 fixes negative-point review by carrying ordered prompt clicks, labels,
point-guided mask refinement, and optional negative-feature refinement through
the Prompt-first and Refine sessions. Outside negative points are ignored by
default and can be re-enabled with `--allow-outside-negative-points`. The
preview-polygon boundary tolerance is controlled by `--negative-point-tolerance`.
Refine preview replacement is now guarded by polygon quality gates such as
point-count, area-ratio, and bbox-overlap checks.

Recommended review command:

```bash
python review_tiles.py \
  --manifest work_dir/tile_auto_austin1/manifests/manifest.json \
  --auto-results work_dir/tile_auto_austin1/auto_results \
  --edited-results work_dir/tile_auto_austin1/edited_results \
  --config configs/prompt_instance_spacenet.json \
  --checkpoint checkpoints/prompt_interactive.pth \
  --device cuda:0 \
  --refine-by-points \
  --point-refine-mode distance \
  --debug-prompt-points \
  --debug-refine-dir work_dir/debug_refine
```

Optional negative-feature refinement:

```bash
python review_tiles.py \
  --manifest work_dir/tile_auto_austin1/manifests/manifest.json \
  --auto-results work_dir/tile_auto_austin1/auto_results \
  --edited-results work_dir/tile_auto_austin1/edited_results \
  --config configs/prompt_instance_spacenet.json \
  --checkpoint checkpoints/prompt_interactive.pth \
  --device cuda:0 \
  --refine-by-points \
  --point-refine-mode distance \
  --negative-feature-refine \
  --negative-similarity-thr 0.75 \
  --negative-feature-patch-size 21 \
  --negative-protect-radius 20 \
  --negative-spatial-sigma 80 \
  --debug-prompt-points \
  --debug-refine-dir work_dir/debug_refine
```
