
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def create_faster_rcnn(num_classes: int = 5, pretrained: bool = True) -> FasterRCNN:
    if pretrained:
        print("Loading pretrained Faster R-CNN (COCO weights)...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
    else:
        print("Initializing Faster R-CNN from scratch...")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    print(f"Model created with {num_classes} classes (including background).")
    return model


def create_faster_rcnn_custom_backbone(num_classes: int = 5, pretrained_backbone=True):
    backbone = torchvision.models.resnet50(weights="IMAGENET1K_V1" if pretrained_backbone else None)
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))  # remove FC layers
    backbone.out_channels = 2048

    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
    )

    model = FasterRCNN(backbone_with_fpn, num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = create_faster_rcnn(num_classes=5)
    dummy = [torch.randn(3, 512, 512)]
    preds = model(dummy)
    print("Output keys:", preds[0].keys())
