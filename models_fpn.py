import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class FPN_Segmenter(nn.Module):
    def __init__(self, num_classes=8):
        super(FPN_Segmenter, self).__init__()

        # Charger le backbone FPN tout prêt
        self.fpn_backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").backbone

        # Tête de segmentation : 1x1 conv pour réduire les canaux
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """Passe avant du modèle FPN + segmentation"""

        # Extraire les features multi-échelles du FPN
        fpn_features = self.fpn_backbone(x)

        # Prendre la feature map de niveau P2 (la plus grande, 128x128)
        p2 = fpn_features['0']

        # Appliquer la convolution finale pour obtenir `num_classes` canaux
        output = self.final_conv(p2)

        # Upsample à la taille de l'image d'entrée (512x512)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=False)

        return output
