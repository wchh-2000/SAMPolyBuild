train_batch_size_per_gpu = 2
train_num_workers = 2
test_batch_size_per_gpu = 6
test_num_workers = 2
persistent_workers = True

backend_args = None
image_size = (1024, 1024)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True,poly2mask=False),
    dict(type='mmdet.Resize', scale=image_size),#将图片、gt框、gt mask缩放到统一尺寸
    dict(type='mmdet.RandomFlip', prob=0.5),
    # dict(type='mmpretrain.ColorJitter',brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    dict(type='mmdet.PackDetInputs')
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=image_size),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
data_parent = 'dataset/whu_mix/'
dataset_type = 'WHUInsSegDataset'

val_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='val/ann.json',
            data_prefix=dict(img_path='val/images'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=4),
            pipeline=test_pipeline,
            backend_args=backend_args))
test_loader = dict(
        batch_size=test_batch_size_per_gpu,
        num_workers=test_num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='test2/ann.json',
            data_prefix=dict(img_path='test2/images'),
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=4),
            pipeline=test_pipeline,
            backend_args=backend_args))
datamodule_cfg = dict(
    type='PLDataModule',
    train_loader=dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        persistent_workers=persistent_workers,
        # shuffle=False,#
        pin_memory=True,
        dataset=dict(
            type=dataset_type,
            data_root=data_parent,
            ann_file='train/ann.json',
            data_prefix=dict(img_path='train/images'),
            filter_cfg=dict(filter_empty_gt=True, min_size=4),
            pipeline=train_pipeline,
            backend_args=backend_args)
    ),
    val_loader=val_loader,
    test_loader=test_loader
    # predict_loader=val_loader
)