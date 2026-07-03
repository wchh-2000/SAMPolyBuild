# SAMPolyBuild
This repository is the code implementation of the paper ["SAMPolyBuild: Adapting the Segment Anything Model (SAM) for Polygonal Building Extraction"](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003563) accepted by ISPRS Journal of Photogrammetry and Remote Sensing. 

![overview](figs/overview.svg)

## Installation
Conda virtual environment is recommended for installation. Please choose the appropriate version of torch and torchvision according to your CUDA version.
Run the following commands or run the 'install.sh' script.
```shell
conda create -n sampoly python=3.10 -y
source activate sampoly # or conda activate sampoly
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
cd pycocotools
pip install .
```
Download the SAM vit_b model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in the 'segment_anything' folder.


## Inference
### Prompt mode
You can use the trained model prompt_interactive.pth from [Baidu Cloud](https://pan.baidu.com/s/1ak-nA032Mf342QHXD8JNug?pwd=8a0q) / [Google Drive](https://drive.google.com/file/d/1meU9SCnXxwAuTYkK0GRtrzRsnAYDgG6B/view?usp=drive_link) to predict the building polygons on the images. Change the **args.imgpth** to the corresponding image path and specify the bounding box prompt coordinates in the **bbox** and prompt point coordinates in the **prompt_point** (selectable).
```shell
python infer_poly_crop.py
```
If you run the code with GUI environment, you can use the following code to interactively select the bounding box and prompt points. Follow the instructions in the terminal to click the bounding box, prompt points (selectable) and generate the polygon.
```shell
python interactive_prompt.py
```
![gui](figs/interactive_gui.png)

### Adaptive bbox utility
The adaptive bbox utility generates a SAMPoly-compatible `xyxy` bbox from one click point without changing the existing prompt inference flow. It can be imported independently:
```python
from utils.auto_bbox import generate_adaptive_bbox

bbox, info = generate_adaptive_bbox(image, (x, y), debug=False)
```
The input `image` is a `numpy.ndarray` in gray, RGB, BGR, or RGBA layout. The function returns `bbox = [x1, y1, x2, y2]` and an `info` dictionary with `success`, `method`, `message`, `click_point`, `roi`, `raw_bbox`, `final_bbox`, `mask_area`, `fallback_used`, and optional `debug_mask` when `debug=True`.

Config values can be passed with a dict or `AutoBBoxConfig`, including `init_window_size`, `min_box_size`, `max_box_size`, `padding`, `floodfill_lo_diff`, `floodfill_up_diff`, `canny_low`, `canny_high`, `morph_kernel_size`, `fallback_box_size`, `min_area_ratio`, `max_area_ratio`, and `max_aspect_ratio`. If adaptive generation fails, the function returns a clipped fixed-size fallback bbox and records the reason in `info["message"]`.

Run the non-GUI smoke test:
```shell
python test_auto_bbox.py --debug
```
Run the image-based debug tool:
```shell
python tools/test_auto_bbox.py --image data/test.png --x 350 --y 220 --out outputs/auto_bbox_debug --debug
```
It saves `bbox_overlay.png`, `info.json`, and `mask.png` when debug mask output is available.

Run the interactive visualization tool:
```shell
python tools/visualize_auto_bbox.py --image figs/eg.jpg --out outputs/auto_bbox_visual --debug
```
Left click on the image to generate an adaptive bbox. Press `c` to clear, `s` to save `bbox_overlay.png`, `info.json`, and optional `mask.png`, or `q`/`Esc` to quit.

Limitations: this utility only generates a bbox. It is not yet connected to `interactive_prompt.py`, `infer_poly_crop.py`, or SAMPoly prediction.

### Auto mode
You can use the trained model auto_whumix.pth from [Baidu Cloud](https://pan.baidu.com/s/1s6aWDZ77t8Bt-aIHiEG9Gw?pwd=6wqn) / [Google Drive](https://drive.google.com/file/d/1VNyUl2CtV19NqxLhnE4LFw32VUvOD0J9/view?usp=drive_link) to predict the building polygons on the images. Change the **args.img_dir** to the image directory that contains the images you want to predict, and the **args.img_suffix** to the corresponding image suffix.
```shell
python infer_auto.py
```
Show the predicted polygons and masks on the images (change the img_dir, dt_pth and img_suffix):
```shell
python utils/show_pred.py
```

### Auto bbox + prompt interaction
Auto mode can be used as a full-image building bbox provider, while prompt mode performs single-building interactive polygon refinement. This does not use `utils/auto_bbox.py`; it reuses the `infer_auto.py` output.

Step 1: run auto mode on the target image directory and save `results.json`.
```shell
python infer_auto.py \
  --config configs/auto_whumix.py \
  --ckpt_path checkpoints/auto_whumix.pth \
  --img_dir data/my_auto_demo/ \
  --img_suffix .png \
  --work_dir work_dir \
  --score_thr 0.1 \
  --gpu 0
```

Step 2: start prompt interaction with the auto results. The first click selects the auto bbox that contains the click point, or the nearest bbox when no bbox contains it. The first click is also used as a positive prompt point. Additional prompt points can be clicked and labeled with `1` for positive or `0` for negative. Press `Enter` to predict and `c` to clear the current interaction.
```shell
python interactive_prompt.py \
  --imgpth data/my_auto_demo/3001001.png \
  --auto_results work_dir/whumix_auto/results.json \
  --auto_image_id 3001001 \
  --auto_min_score 0.1 \
  --gpu 0
```

If `--auto_image_id` is omitted, the image file stem is used. For example, `3001001.png` maps to image id `3001001`.

Prompt interaction saves the latest per-building result to:
```text
work_dir/prompt_instance_spacenet/interactive_results.json
work_dir/prompt_instance_spacenet/interactive_masks/
```
Each selected building uses a stable instance key. Re-predicting the same selected bbox overwrites that instance's `latest_polygon`, `click_points`, `latest_mask_path`, and `version`. Results for other selected buildings are kept. The display removes the previous polygon for the same building before drawing the latest one.

Use `--debug_prompt_points` with `interactive_prompt.py` to print raw click points, crop-transformed prompt points, predictor-transformed points, and labels before prompt encoding and mask prediction.

Use `--refine_by_points` to run point-guided mask post-processing before polygon extraction. This does not change the model. The default `--point_refine_mode connected_component` removes connected mask components that contain negative prompt points and keeps components that contain positive prompt points. Use `--point_refine_mode distance` when the target building and negative building are stuck in one connected component; pixels closer to negative points are removed, and `--negative_margin` can be increased to remove more area near negative points. The latest debug masks are saved to:
```text
work_dir/prompt_instance_spacenet/mask_before_point_refine.png
work_dir/prompt_instance_spacenet/mask_after_point_refine.png
work_dir/prompt_instance_spacenet/mask_point_refine_regions.png
```

Use `--negative_feature_refine` to run negative-feature mask post-processing before polygon extraction. This does not change the model and is disabled by default. Each negative prompt point builds an RGB/HSV/LAB patch prototype, searches visually similar pixels inside the current predicted mask, protects positive prompt neighborhoods, and then extracts the polygon from the refined mask. Lower `--negative_similarity_thr` removes more pixels; higher values are more conservative. Larger `--negative_spatial_sigma` lets the negative feature affect a wider area, and `--negative_spatial_sigma 0` disables spatial weighting. The latest debug outputs are saved to:
```text
work_dir/prompt_instance_spacenet/negative_patch.png
work_dir/prompt_instance_spacenet/negative_similarity_map.png
work_dir/prompt_instance_spacenet/mask_before_refine.png
work_dir/prompt_instance_spacenet/mask_after_refine.png
work_dir/prompt_instance_spacenet/polygon_before_refine.png
work_dir/prompt_instance_spacenet/polygon_after_refine.png
```
Recommended first-pass parameters:
```shell
python interactive_prompt.py \
  --imgpth data/my_auto_demo/3001001.png \
  --auto_results work_dir/whumix_auto/results.json \
  --auto_image_id 3001001 \
  --auto_min_score 0.1 \
  --negative_feature_refine \
  --negative_similarity_thr 0.75 \
  --negative_feature_patch_size 21 \
  --negative_protect_radius 20 \
  --negative_spatial_sigma 80 \
  --gpu 0
```
Limitations: this first version uses hand-crafted color statistics rather than deep image features. It works best for tree, shadow, grass, or adjacent-roof regions with visual contrast. If roof pixels are removed, raise `--negative_similarity_thr`, reduce `--negative_spatial_sigma`, or increase `--negative_protect_radius`.

Run the non-GUI negative feature refinement check:
```shell
python tools/test_negative_feature_refine.py \
  --out_dir work_dir/negative_feature_refine_debug
```
Expected output includes `Negative feature refinement applied`, fewer `after_pixels` than `before_pixels`, `protected_positive_kept=True`, and `negative_region_removed=True`.

Run the non-GUI bbox selection check:
```shell
python tools/test_select_auto_bbox.py \
  --results work_dir/whumix_auto/results.json \
  --image data/my_auto_demo/3001001.png \
  --x 520 \
  --y 320 \
  --min-score 0.1
```

Expected output is a JSON object with `selected_bbox_xyxy`, `contains_click`, `score_cls`, and distance fields. If no candidate is found, check that the image id matches `results.json`, lower `--min-score`, or run `infer_auto.py` again for the target image.

Limitations: auto bbox selection depends on an existing `infer_auto.py` result file and checkpoint. It does not run the auto detector inside `interactive_prompt.py`, and it does not modify the auto detection network or training flow.

### Headless single-click prompt inference

`tools/infer_single_click.py` provides a non-GUI prompt inference entry point for external applications. It does not import Matplotlib, wait for mouse or keyboard events, or change the behavior of `interactive_prompt.py`. The command-line click and the selected bbox are passed to the existing `SamPredictor`, and the decoded polygon is converted back to original-image pixel coordinates.

Fixed bbox example:
```shell
python tools/infer_single_click.py \
  --imgpth data/my_auto_demo/3001001.png \
  --click_x 500 \
  --click_y 400 \
  --work_dir work_dir \
  --output_json work_dir/single_click_result.json \
  --checkpoint prompt_interactive.pth \
  --config configs/prompt_instance_spacenet.json \
  --bbox_mode fixed \
  --bbox_size 256 \
  --gpu 0
```

Auto bbox example:
```shell
python tools/infer_single_click.py \
  --imgpth data/my_auto_demo/3001001.png \
  --click_x 500 \
  --click_y 400 \
  --work_dir work_dir \
  --output_json work_dir/single_click_result_auto.json \
  --checkpoint prompt_interactive.pth \
  --config configs/prompt_instance_spacenet.json \
  --bbox_mode auto \
  --bbox_size 256 \
  --auto_results work_dir/whumix_auto/results.json \
  --auto_image_id 3001001 \
  --auto_min_score 0.1 \
  --gpu 0
```

`--bbox_mode fixed` creates a boundary-clipped square centered on the click. `--bbox_mode auto` selects an auto detection only when it contains the click; missing results, invalid results, or a nearest bbox that does not contain the click fall back to the fixed bbox. The default prompt checkpoint is `prompt_interactive.pth`, the default config is `configs/prompt_instance_spacenet.json`, and `--gpu -1` selects CPU inference.

Successful output contains `success`, `image_path`, `click`, `bbox`, `bbox_source`, and one instance with `score` and `latest_polygon`. Failures return a nonzero exit code and attempt to write the same JSON envelope with `success: false`, an empty `instances` list, and the error message.

Run the non-GUI utility tests before model inference:
```shell
python tools/test_single_click_headless.py
python tools/infer_single_click.py --help
```

Limitations: the checkpoint must use the Lightning-style prompt checkpoint format expected by `build_sam(load_pl=True)`. Auto mode reads an existing auto `results.json`; it does not run the auto detector. The prompt input preserves the existing interactive behavior by using the bbox center and command-line click as positive points.

### Reusable prompt inference service

`services.prompt_inference_service.PromptInferenceService` provides a stateless Python API for CLI, desktop, test, or future server adapters. It does not parse command-line arguments, write JSON, open a GUI, or depend on FastAPI. The model is loaded once by `initialize()` and reused by subsequent `infer()` calls.

```python
from services.models import InferenceRequest, PromptPoint, PromptServiceConfig
from services.prompt_inference_service import PromptInferenceService

service = PromptInferenceService(
    PromptServiceConfig(
        checkpoint="prompt_interactive.pth",
        model_config="configs/prompt_instance_spacenet.json",
        gpu=0,
        bbox_mode="auto",
        bbox_size=256,
    )
)
service.initialize()

response = service.infer(
    InferenceRequest(
        image_path="data/my_auto_demo/3001001.png",
        prompts=[
            PromptPoint(x=500, y=400, label=1),
            PromptPoint(x=620, y=430, label=0),
        ],
        auto_results="work_dir/whumix_auto/results.json",
        auto_image_id="3001001",
    )
)
print(response.to_dict())
service.close()
```

Prompt coordinates and optional request bboxes use the original image pixel coordinate system. An explicit request bbox has highest priority. Otherwise, auto mode selects a detection containing the first positive point and falls back to a boundary-clipped fixed bbox. When `include_bbox_center_prompt=True`, the bbox center is added as an extra positive point to preserve existing interactive behavior. The returned polygon is converted back to original-image coordinates and validated for point count, finite coordinates, closure, and nonzero area.

Run the Linux service checks:
```shell
conda activate sampoly

python -c "from services.prompt_inference_service import PromptInferenceService; print('ok')"
python tools/test_prompt_inference_service.py --help

python tools/test_prompt_inference_service.py \
  --imgpth path/to/test.png \
  --checkpoint path/to/prompt_interactive.pth \
  --config configs/prompt_instance_spacenet.json \
  --click_x 500 \
  --click_y 400 \
  --bbox_mode fixed \
  --gpu 0

python tools/test_prompt_inference_service.py \
  --imgpth path/to/test.png \
  --checkpoint path/to/prompt_interactive.pth \
  --config configs/prompt_instance_spacenet.json \
  --click_x 500 \
  --click_y 400 \
  --neg_x 620 \
  --neg_y 430 \
  --bbox_mode fixed \
  --gpu 0
```

The initial service is stateless: each `infer()` call reloads the image and recomputes its embedding. It serializes access to the predictor because the predictor stores mutable image features. Session and embedding reuse are intentionally deferred to a later stage.

## Dataset Preparation
### SpaceNet Vegas Dataset
We converted the original images of the SpaceNet dataset to 8-bit and the annotations to coco format, and divided them into training, validation, and test sets in the ratio of 8:1:1, which are available for download from [here](https://aistudio.baidu.com/datasetdetail/269168). Place the train, val, test folders in the 'dataset/spacenet' folder.
### WHU-mix (vector) dataset
The WHU-mix dataset can be download from [here](http://gpcv.whu.edu.cn/data/whu-mix%20(vector)/whu_mix(vector).html). Place the train, val, test1 and test2 folders in the 'dataset/whu_mix' folder, and run the preprocess code:
```shell
cd dataset
python preprocess.py
```

### Custom Dataset
The custom dataset should be in the following format, or change the **train_dataset_pth**, **val_dataset_pth** in the train.py and **dataset_pth** in the test.py to the corresponding path.
```
dataset
├── dataset_name
    ├── train
    |    ├── images
    |    ├── ann.json
    ├── val
    |    ├── images
    |    ├── ann.json
    ├── test
        ├── images
        ├── ann.json
```

## Training
### Prompt mode
Single gpu:
```shell
python train.py --config configs/prompt_instance_spacenet.json --gpus 0
```
Multi gpus:
```shell
python train.py --config configs/prompt_instance_spacenet.json --gpus 0 1 --distributed
```
### Auto mode
First pretrain the model with the full-image feature input (the default method of the SAM) in prompt mode.
```shell
python train.py --config configs/prompt_fullimg_spacenet.json
```
Then load the model and train the auto mode. Change the pretrain_chkpt in the auto_spacenet.py to the path of the pretrained model.
```shell
python train_auto.py --config configs/auto_spacenet.py
```

## Testing
### Prompt mode
Evaluate the model on the test set, and save the results:
```shell
python test.py
```
You need to change the **--task_name** to the corresponding training task name, and the other arguments will be set automatically according to training configuration.

If you want to use our trained model to evaluate, you can download prompt_instance_spacenet.pth from [Baidu Cloud](https://pan.baidu.com/s/1xQ3tKt2mOv55O0g3J-EJvQ?pwd=dz5d) / [Google Drive](https://drive.google.com/file/d/1pQ_1HmUfCpJ_c6LZ3qcvbhabH6CpBgp4/view?usp=drive_link) and change the following code in the test.py:
```python
args = load_args(parser,path='configs/prompt_instance_spacenet.json')
args.checkpoint = 'prompt_instance_spacenet.pth'
```
### Auto mode
Set the **config** and **ckpt_path** args to the corresponding configuration and checkpoint path, and run the test.
```shell
python test_auto.py --config configs/auto_spacenet.py --ckpt_path work_dir/spacenet_auto/version_0/checkpoints/last.ckpt
```
You can download the trained model auto_spacenet.pth from [Baidu Cloud](https://pan.baidu.com/s/1AIvaoI-hM0Ecd94S_sag4w?pwd=in3k) / [Google Drive](https://drive.google.com/file/d/1oJ2Pmip3B60lFSStIOeoAwnG0__YK29-/view?usp=drive_link) and test directly.
```shell
python test_auto.py --config configs/auto_spacenet.py --ckpt_path auto_spacenet.pth
```
For the WHU-mix dataset, you can download the trained model auto_whumix.pth from [Baidu Cloud](https://pan.baidu.com/s/1s6aWDZ77t8Bt-aIHiEG9Gw?pwd=6wqn) / [Google Drive](https://drive.google.com/file/d/1VNyUl2CtV19NqxLhnE4LFw32VUvOD0J9/view?usp=drive_link) and test directly.

For test2:
```shell
python test_auto.py --config configs/auto_whumix.py --ckpt_path auto_whumix.pth --gt_pth dataset/whu_mix/test2/ann.json
```
For test1: (change **test2** in configs/data_whu_mix.py to **test1**)
```shell
python test_auto.py --config configs/auto_whumix.py --ckpt_path auto_whumix.pth --gt_pth dataset/whu_mix/test1/ann.json --score_thr 0.3
```
## Acknowledgement
This project is developed based on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
 and [RSPrompter](https://github.com/KyanChen/RSPrompter) project. We thank the authors for their contributions.

## Citation
If you use the code of this project in your research, please refer to the bibtex below to cite SAMPolyBuild.
```
@article{wang2024sampolybuild,
  title={SAMPolyBuild: Adapting the Segment Anything Model for polygonal building extraction},
  author={Wang, Chenhao and Chen, Jingbo and Meng, Yu and Deng, Yupeng and Li, Kai and Kong, Yunlong},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={218},
  pages={707--720},
  year={2024},
  publisher={Elsevier},
  doi = {10.1016/j.isprsjprs.2024.09.018}
}
```
## License

This project is licensed under the [Apache 2.0 license](LICENSE).

## Contact
If you have any questions, please contact wangchenhao22@mails.ucas.ac.cn.
