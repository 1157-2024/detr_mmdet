from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载配置文件和权重
config_file = 'configs/detr/detr_r50_xjh.py'
checkpoint_file = 'detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# COCO 数据集的类别名称
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

colors = np.random.randint(0, 255, size=(len(class_names), 3)).astype(int)
# 进行推理
img = '1.jpg'  # 你的图片路径
result = inference_detector(model, img)

# 加载图片
image = cv2.imread(img)

# 可视化结果
# 假设 result 已经是 DetDataSample 对象
# 直接访问 bboxes, scores 和 labels
bboxes = result.pred_instances.bboxes  # 获取边界框
scores = result.pred_instances.scores  # 获取置信度分数
labels = result.pred_instances.labels  # 获取类别标签

# 绘制每一个检测结果
for i in range(len(bboxes)):
    if scores[i] > 0.7:  # 只处理高置信度的框
        # x1, y1, x2, y2 = bboxes[i].int()  # 使用 .int() 转换为整数类型。
        # 转换 Tensor 为整数坐标
        x1, y1, x2, y2 = int(bboxes[i][0].item()), int(bboxes[i][1].item()), int(bboxes[i][2].item()), int(bboxes[i][3].item())

        class_id = labels[i].item()  # 确保 class_id 是整数
        color = colors[class_id].tolist()  # 获取对应类别的颜色
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 准备文本
        text = f'{class_names[class_id]}: {scores[i]:.2f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # 绘制文本背景
        cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        # 绘制文本
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 保存结果
output_file = 'output_image.jpg'  # 输出文件路径
cv2.imwrite(output_file, image)

