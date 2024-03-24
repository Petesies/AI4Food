# The new config inherits a base config to highlight the necessary modification
_base_ = '../../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=2)

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))  #NUm classes 103 w/ background, 102 w/o

# Modify dataset related settings
data_root = "C:/Users/peter/mmdetection/configs/AI4Food/configs/data/stickers/"
metainfo = {    #Remove background once regenerated annotations done as only instance segmentation needed, not semantic
    'classes': ('stickers'),
    'palette': [
        (90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60),(90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60),(90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60),(90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60),(90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60),(90, 90, 90),(150, 50, 10),(100, 30, 60),(100, 20, 40),(20, 220, 120),(90, 20, 40),(10, 60, 20),(210, 20, 40),(180, 10, 20),(120, 200, 50),(20, 0, 160),(20, 90, 30),(130, 10, 40),(150, 20, 60),(20, 10, 60),(20, 60, 60),(20, 30, 60),(0, 90, 90),(20, 0, 70),(20, 80, 20),(210, 230, 60),(20, 230, 60),(2, 20, 60),(1, 20, 60),(140, 30, 20),(220, 220, 10),(210, 40, 80),(20, 20, 20),(120, 120, 120),(20, 20, 30),(10, 210, 60),(120, 20, 10),(220, 200, 60),(220, 20, 160),(20, 20, 60),(120, 20, 60)
    ]
}
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/train.json',
        data_prefix=dict(img='train/')),
    batch_size=1)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# optim_wrapper = dict(  # Optimizer wrapper config
#     type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
#     optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
#         type='Adam',  # Stochastic gradient descent optimizer
#         lr=0.001,  # The base learning rate
#         weight_decay=0.0001),  # Weight decay of SGD
#     # clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
#     )
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    # Parameter-level learning rate and weight decay settings
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        },
        norm_decay_mult=0.0),

    # gradient clipping
    clip_grad=dict(max_norm=0.01, norm_type=2))
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='Adam', lr=0.00000001, weight_decay=0.0001))
# momentum=0.09,  # Stochastic gradient descent with momentum
#LR reduce
#change optimizer ex. ADAM
#different models
#mmdet optimizers lookup submodule
#Look into SAM or FastSAM etc.
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     by_epoch=True,
#     warmup_iters=1,
#     warmup_ratio=0.001,
#     step=[7])

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/val.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'




