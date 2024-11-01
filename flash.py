import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
from mmdet.apis import init_detector, inference_detector

app = Flask(__name__)

# 初始化模型
config_file = 'configs/detr/detr_hyperparameters.py'
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

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        # 处理上传的文件
        file = request.files['image']
        if file:
            file_path = 'uploaded_image.jpg'
            file.save(file_path)
            
            # 进行目标检测
            result = inference_detector(model, file_path)
            image = cv2.imread(file_path)
            
            # 可视化检测结果
            bboxes = result.pred_instances.bboxes
            scores = result.pred_instances.scores
            labels = result.pred_instances.labels
            
            for i in range(len(bboxes)):
                if scores[i] > 0.6:  # 设置置信度阈值
                    x1, y1, x2, y2 = int(bboxes[i][0].item()), int(bboxes[i][1].item()), int(bboxes[i][2].item()), int(bboxes[i][3].item())
                    class_id = labels[i].item()
                    color = colors[class_id].tolist()
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # 准备标签文字
                    text = f'{class_names[class_id]}: {scores[i]:.2f}'
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 绘制文本背景和文字
                    cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 保存并返回可视化结果
            output_file = 'output_image.jpg'
            cv2.imwrite(output_file, image)
            return send_file(output_file, mimetype='image/jpeg')
    
    # GET 请求，显示上传页面
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
