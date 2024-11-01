_base_ = 'detr_r50_8xb2-150e_coco.py'

model = dict(
    # 模型参数不在此脚本修改
)

# ------------------------------------- #
#  更换数据集修改参数 
# ------------------------------------- #
dataset_type = 'CocoDataset'
data_root = '/root/datasets/COCO2017'
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json', # annotation
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
))
test_dataloader = val_dataloader
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
)
test_evaluator = val_evaluator


load_from = 'detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'