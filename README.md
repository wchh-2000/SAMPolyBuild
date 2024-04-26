# SAMPolyBuild
Adapting the Segment Anything Model (SAM) for Polygonal Building Extraction

## Installation
Conda virtual environment is recommended for installation. Please choose the appropriate version of torch and torchvision according to your CUDA version.
```shell
conda create -n sampoly python=3.10 -y
source activate sampoly # or conda activate sampoly
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
Download the SAM vit_b model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it in the 'segment_anything' folder.

## Dataset Preparation
### SpaceNet Vegas Dataset
We converted the original images of the SpaceNet dataset to 8-bit and the annotations to coco format, and divided them into training, validation, and test sets in the ratio of 8:1:1, which are available for download from [here](https://aistudio.baidu.com/datasetdetail/269168). Place the train, val, test folders in the 'dataset/spacenet' folder.

Crop the image according to the bbox of building instances and obtain the annotations of the instances' polygonal contours:
```shell
python dataset/crop_bbox.py
```

## Training
Single gpu:
```shell
python train.py --config configs/prompt_instance_spacenet.json --gpus 0
```
Multi gpus:
```shell
python train.py --config configs/prompt_instance_spacenet.json --gpus 0 1 --distributed
```

## Testing
Evaluate with pre-cropped images:
```shell
python test.py
```
Given a file containing bbox detections or annotations, crop the image with the bbox and predict the polygon results which are restored to the original image coordinates:
```shell
python test.py --predict_crop_input --img_dir dataset/spacenet/test/images --bbox_file dataset/spacenet/test/ann.json
```
Notes that the bbox file should be in coco format, and don't need to contain the 'segmentation' field.

## Acknowledgement
This project is developed based on the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) project. We thank the authors for their contributions.

## License

This project is licensed under the [Apache 2.0 license](LICENSE).

## Contact
If you have any questions, please contact wangchenhao22@mails.ucas.ac.cn.
