freeze_imgEncoder=False
add_pe=False
add_edge=True
center_prompt=True
expand_roi=True
expand_scale=0.1
roi_size=24#roi特征图大小，原始SAM特征图大小64  24
selected_channels=range(1, 13, 2)
debug=False
if debug:
    use_distributed_sampler=False
    gpus=[0]
    task_name = 'debug'
else:
    use_distributed_sampler=True
    gpus=[0,1]
    task_name = 'spacenet_auto'

max_epochs = 50
log_root=f'/data/work_dir/Prompter/{task_name}/'
logger=dict(
    type='TensorBoardLogger',
    save_dir='/data/work_dir/Prompter/',
    name=task_name
)
param_scheduler_callback = dict(
    type='ParamSchedulerHook'
)
# train:
callbacks = [
    param_scheduler_callback,
    dict(
        type='ModelCheckpoint',
        # dirpath=log_root+'checkpoints',
        save_last=True,
        mode='max',
        monitor='valmask_iou',
        save_top_k=3,
        filename='{epoch}-{valmask_iou:.2f}'
    ),
    dict(
        type='LearningRateMonitor',
        logging_interval='step'
    )
]
vis_backends = [dict(type='mmdet.LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
    fig_save_cfg=dict(
    frameon=True,
    figsize=(5,5),
    # dpi=300,
    ),
    line_width=2,
    alpha=0.8)
trainer_cfg = dict(
    compiled_model=False,
    accelerator="gpu",
    # strategy="auto",
    strategy='ddp_find_unused_parameters_true',
    # precision='32',
    # precision='16-mixed',
    devices=gpus,
    default_root_dir=log_root,
    max_epochs=max_epochs,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=20,
    check_val_every_n_epoch=1,
    benchmark=True,
    # val_check_interval=None,
    num_sanity_val_steps=0,
    use_distributed_sampler=use_distributed_sampler,
)
if debug:
    trainer_cfg.update(
        dict(limit_train_batches=30,
            limit_val_batches=30))
    
sub_model_train = [
    'panoptic_head',
    'data_preprocessor',
    'backbone.mask_decoder',
    # 'backbone.image_encoder'
]#backbone 为sam的所有参数不训练

sub_model_optim = {
    'panoptic_head': {'lr_mult': 1},
    'backbone.mask_decoder':{'lr_mult': 0.1},
    # 'backbone.image_encoder':{'lr_mult': 0.05},
}
if not freeze_imgEncoder:
    sub_model_train.append('backbone.image_encoder')
    sub_model_optim['backbone.image_encoder'] ={'lr_mult': 0.01}
optimizer = dict(
    type='AdamW',
    sub_model=sub_model_optim,
    lr=0.0005,
    weight_decay=1e-4
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    ),
]
evaluator_ = dict(
        type='CocoPLMetric',
        metric=['bbox', 'mask','poly'],
        proposal_nums=[100, 300, 500],
        cls_thr=0.1
)

evaluator = dict(
    val_evaluator=evaluator_,
)

data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
)

num_things_classes = 1
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

model_cfg = dict(
    type='SegSAMAnchorPLer',
    hyperparameters=dict(
        optimizer=optimizer,
        param_scheduler=param_scheduler,
        evaluator=evaluator,
    ),
    need_train_names=sub_model_train,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # type='vit_h',
        # checkpoint='/data/pretrained/sam_vit_h_4b8939.pth',
        model_type='vit_b',
        checkpoint='work_dir/prompt_fullimg_spacenet/version_0/checkpoints/last.ckpt',
        load_pl=True,
        selected_imgfeats=selected_channels,
        add_edge=add_edge,
        upconv=True,
        # type='vit_l',
        # checkpoint='/data/pretrained/sam_vit_l_0b3195.pth',        
    ),
    panoptic_head=dict(
        type='SAMAnchorInstanceHead',#继承TwoStageDetector
        weight=dict(bbox=6.5,cls=10),
        neck=dict(
            type='RSFPN',
            feature_aggregator=dict(
                type='RSFeatureAggregator',
                model_arch='base',#'base','large','huge'
                out_channels=256,
                hidden_channels=32,
                select_layers=selected_channels,
            ),
            feature_spliter=dict(
                type='RSSimpleFPN',
                backbone_channel=256,
                in_channels=[64, 128, 256, 256],
                out_channels=256,
                num_outs=5,
                norm_cfg=dict(type='LN2d', requires_grad=True)),
        ),
        rpn_head = dict(
            type='mmdet.RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='mmdet.AnchorGenerator',
                scales=[6, 10,16],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32,64]),
            bbox_coder=dict(
                type='mmdet.DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            reg_decoded_bbox=True,#for GIoU
            loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=10.0)),
        roi_head=dict(
            type='SAMAnchorPromptRoIHead',#继承StandardRoIHead
            expand_roi=expand_roi,
            expand_scale=expand_scale,
            add_pe=add_pe,
            bbox_roi_extractor = dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_roi_extractor=dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[16]),     
            bbox_head = dict(
                type='mmdet.Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='mmdet.DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                reg_decoded_bbox=True,#for GIoU
                loss_bbox=dict(type='mmdet.GIoULoss', loss_weight=10.0)),
            mask_head=dict(
                type='SAMPromptMaskHead',
                loss_weight=dict(vmap=0.5,voff=3.5,mask=0.7,edge=0.3),
                expand_roi=expand_roi,
                expand_scale=expand_scale,
                center_prompt=center_prompt,
                add_edge=add_edge,
                loss_mask=dict(#没用
                    type='mmdet.CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,#Upper bound number of negative and positive samples
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(#self.rpn_head.loss_and_predict( 结果作为roi_head的输入
                nms_pre=2000,# nms 前每个输出层最多保留预测score较大的前nms_pre个预测框，n个特征输出层总共n*nms_pre个预测框
                max_per_img=1000,#nms的最大输出个数
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=2),
            rcnn=dict(#对应roi_head
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict( #bbox_head的roi sampler
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,#num*pos_fraction个正样本
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                roi_mask_sampler=dict(#mask_head的roi sampler
                    type='mmdet.RandomSampler',
                    num=64,
                    pos_fraction=1,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=roi_size*4,#区域mask标签的尺寸
                pos_weight=-1,
                debug=False),debug=debug),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),#RPNHead内有nms
                min_bbox_size=2),
            rcnn=dict(#对应roi_head bbox_head中nms
                score_thr=0.1,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,# 最终输出的每张图片最多实例个数
                mask_thr_binary=0.5)
        )
    )
)