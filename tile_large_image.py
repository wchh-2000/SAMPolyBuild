"""Root CLI compatibility entry point for the tile workflow.

The implementation remains in ``tools.tile_large_image`` so there is only one
tiling code path to maintain.
"""

from tools.tile_large_image import main


if __name__ == "__main__":
    raise SystemExit(main())
