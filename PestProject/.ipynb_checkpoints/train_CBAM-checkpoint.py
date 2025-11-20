# 猴子补丁部分 (保持不变)
from custom_modules import CBAM
import ultralytics.nn.tasks
ultralytics.nn.tasks.CBAM = CBAM
from ultralytics import YOLO

def main():
    # 指定自定义模型配置文件
    model_config_path = 'custom_yolov11n_cbam.yaml'
    # 指定数据集配置文件
    data_config_path = '/root/autodl-tmp/dataset/data.yaml'

    # *** 关键修改在这里 ***
    # 我们不是加载 'yolov8n.pt'，而是加载我们的 yaml 文件来构建模型结构。
    # 然后，我们再手动加载预训练权重到这个模型中。
    # 注意：这里我们先构建一个空的自定义结构模型
    model = YOLO(model_config_path)

    # 手动加载权重。这会把 yolov8n.pt 中与我们模型结构匹配的层的权重加载进来
    # 这是一种更高级的用法，可以避免网络下载并利用预训练知识
    model.load('yolo11n.pt') 

    print("自定义模型结构创建并加载预训练权重成功，开始训练...")
    
    results = model.train(
        data=data_config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8n_cbam_pest_detection_finetuned'
    )
    
    print("训练完成！")

if __name__ == '__main__':
    main()