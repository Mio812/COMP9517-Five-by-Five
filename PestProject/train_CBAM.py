
from custom_modules import CBAM
import ultralytics.nn.tasks
ultralytics.nn.tasks.CBAM = CBAM
from ultralytics import YOLO

def main():
    
    model_config_path = 'custom_yolov11n_cbam.yaml'
    data_config_path = '/root/autodl-tmp/dataset/data.yaml'

    
    model = YOLO(model_config_path)

    model.load('yolo11n.pt') 

    print("Start！")
    
    results = model.train(
        data=data_config_path,
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8n_cbam_pest_detection_finetuned'
    )
    
    print("complete！")

if __name__ == '__main__':
    main()
