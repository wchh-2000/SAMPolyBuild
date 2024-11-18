_base_=['./bbox_prompter.py','./data_whu_mix.py']
debug=False
pretrain_chkpt='work_dir/prompt_fullimg_whumix/version_0/checkpoints/last.ckpt'
if debug:
    task_name = 'debug'
    gpus=[0]
    use_distributed_sampler=False
else:
    task_name = 'whumix_auto'
    gpus=[0,1]
    use_distributed_sampler=True

max_epochs = 24
log_every_n_steps = 20
eval_interval = 2
log_root=f'work_dir/{task_name}/'
logger=dict(
    type='TensorBoardLogger',
    save_dir='work_dir/',
    name=task_name
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
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        by_epoch=True,
        begin=1,
        end=max_epochs,
    ),
]
trainer_cfg = dict(#覆盖_base_中的trainer_cfg
    devices=gpus,#
    default_root_dir=log_root,
    max_epochs=max_epochs,
    logger=logger,
    log_every_n_steps=log_every_n_steps,
    check_val_every_n_epoch=eval_interval,
    num_sanity_val_steps=2,
    use_distributed_sampler=use_distributed_sampler,
)
model_cfg = dict(debug=debug,
            hyperparameters=dict(param_scheduler=param_scheduler),
            backbone=dict(checkpoint=pretrain_chkpt)
        )
if debug:
    trainer_cfg.update(
        dict(limit_train_batches=12,
            limit_val_batches=4))