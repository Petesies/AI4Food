# The new config inherits a base config to highlight the necessary modification
_base_ = '../ms_rcnn/ms-rcnn_x101-64x4d_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=33), mask_head=dict(num_classes=33)))

# Modify dataset related settings
data_root = 'C:/Users/peter/mmdetection/configs/food/data/Ingredients/'
metainfo = {
    'classes': ('Apple', 'Basmatirice', 'Blackpepper', 'Broccoli', 'Brownsugar', 'Butter', 'Buttermilk', 'Buttonmushroom', 'Cashewnut', 'Chickenstock', 'Cilantro', 'Cinnamon', 'Egg', 'Flour', 'Garlic', 'Greenpepper', 'Lemon', 'Mayonnaise', 'Medjooldates', 'milk', 'Mustard', 'null', 'Onion', 'Peas', 'Potato', 'Redbeans', 'Redpepper', 'Salt', 'Springonion', 'Tomato', 'Vegetableoil', 'Whitesugar', 'yeast', ),
    'palette': [ #NEED TO EXPAND TO 33, CURRENTLY 13
        (220, 20, 60),(200, 20, 160),(20, 20, 60),(20, 20, 160),(220, 120, 160),(20, 120, 160),(20, 210, 160),(20, 60, 90),(120, 120, 160),(20, 220, 160),(210, 130, 160), (220, 20, 60),(200, 20, 160),(20, 20, 60),(20, 20, 160),(220, 120, 160),(20, 120, 160),(20, 210, 160),(20, 60, 90),(120, 120, 160),(20, 220, 160),(210, 130, 160), (220, 20, 60),(200, 20, 160),(20, 20, 60),(20, 20, 160),(220, 120, 160),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco/ms_rcnn_x101_64x4d_fpn_1x_coco_20200206-86ba88d2.pth'