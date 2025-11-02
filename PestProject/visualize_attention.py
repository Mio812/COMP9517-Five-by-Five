from custom_modules import CBAM
import ultralytics.nn.tasks
ultralytics.nn.tasks.CBAM = CBAM
import cv2
import torch
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'runs/detect/yolov8n_cbam_pest_detection/weights/best.pt'
IMAGE_PATH = '/root/autodl-tmp/dataset/test/images/ants-17-_jpg.rf.366ce3d542821626b2926e3142d1bb64.jpg'
CBAM_LAYER_INDEX = 16

activation_map = None

def get_activation_hook(module, input, output):
    global activation_map
    activation_map = output.detach()

def main():
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully.")

    try:
        target_layer = model.model[CBAM_LAYER_INDEX]
        if isinstance(target_layer, CBAM):
            handle = target_layer.register_forward_hook(get_activation_hook)
            print(f"Successfully registered hook on layer {CBAM_LAYER_INDEX} ({type(target_layer).__name__}).")
        else:
            print(
                f"Error: Layer {CBAM_LAYER_INDEX} is not a CBAM module, but {type(target_layer).__name__}. Please check the model definition file and index.")
            return
    except IndexError:
        print(f"Error: Index {CBAM_LAYER_INDEX} is out of the model's layer range. Please check.")
        return

    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"Error: Cannot read image {IMAGE_PATH}")
        return

    original_height, original_width = original_img.shape[:2]

    print("Performing inference to capture the activation map...")
    model.predict(IMAGE_PATH, verbose=False)

    handle.remove()
    print("Hook has been removed.")

    if activation_map is not None:
        print(f"Captured activation map shape: {activation_map.shape}")

        heatmap = activation_map.cpu().numpy()[0, 0, :, :]
        
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-6)
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.resize(heatmap, (original_width, original_height))

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)

        output_path = 'attention_visualization.jpg'
        cv2.imwrite(output_path, superimposed_img)
        print(f"Attention visualization result saved to: {output_path}")
    else:
        print("Error: Failed to capture the activation map.")

if __name__ == '__main__':
    main()
