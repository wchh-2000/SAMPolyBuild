import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import copy
import mmengine.fileio as fileio
from mmpl.registry import DATASETS
from mmengine.dataset import BaseDataset, Compose

@DATASETS.register_module()
class PredictDataset(BaseDataset):
    """Custom dataset for image prediction without annotations.

    Example file structure:

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx.jpg
        │   │   │   │   ├── yyy.jpg
        │   │   │   │   ├── zzz.jpg
        │   │   │   ├── val

    The `BasePredictDataset` only loads image files from `img_dir`.

    Args:
        img_suffix (str): Suffix of images. Default: '.jpg'.
        data_root (str, optional): The root directory for `data_prefix`. Defaults to None.
        data_prefix (dict, optional): Prefix for image data. Defaults to dict(img_path='').
        filter_cfg (dict, optional): Config for filtering data. Defaults to None.
        indices (int or Sequence[int], optional): Use a subset of data for prediction.
            Defaults to None, which means using all data.
        serialize_data (bool, optional): Whether to hold memory using serialized objects.
            Defaults to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): Should always be True for prediction. Defaults to True.
        lazy_init (bool, optional): Whether to delay loading annotations. Not applicable here,
            but kept for compatibility. Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """
    METAINFO: dict = dict()

    def __init__(self,
                 img_suffix: str = '.jpg',
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = True,
                 lazy_init: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.ann_file=''
        self.img_suffix = img_suffix
        self.backend_args = backend_args.copy() if backend_args else None

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.lazy_init = lazy_init
        self.data_list: List[dict] = []

        # Set meta information.
        self._metainfo = self._load_metainfo(copy.deepcopy(None))  # No metainfo needed


        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Initialize the dataset.
        if not self.lazy_init:
            self.full_init()

    def load_data_list(self) -> List[dict]:
        """Load image data from the specified directory.

        Returns:
            list[dict]: All data info of the dataset containing only image paths.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        if img_dir is None:
            raise ValueError("`img_path` must be specified in `data_prefix`.")

        if self.data_root is not None:
            img_dir = osp.join(self.data_root, img_dir)

        # Ensure the image directory exists
        if not osp.isdir(img_dir):
            raise FileNotFoundError(f"Image directory '{img_dir}' does not exist.")

        # List all image files with the specified suffix
        for idx, img in enumerate(sorted(fileio.list_dir_or_file(
                dir_path=img_dir,
                list_dir=False,
                suffix=self.img_suffix,
                recursive=True,
                backend_args=self.backend_args))):
            img_path = osp.join(img_dir, img)
            
            # Use the filename without extension as img_id
            img_id = osp.splitext(img)[0]
            # img_id=idx

            data_info = dict(
                img_path=img_path,
                img_id=img_id  # Add img_id to the data_info dictionary
            )
            # data_info['seg_fields'] = []  # No segmentation fields
            data_list.append(data_info)

        return data_list