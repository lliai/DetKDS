from .detection_distillation_faster_rcnn import DetectionDistiller_FR
from .detection_distillation_cm import DetectionDistiller_CM
from .detection_distillation_reppoints import DetectionDistiller_Rep
from .detection_distillation_retinanet import DetectionDistiller_Retina
from .detection_distillation_fcos import DetectionDistiller_FCOS
from .detection_distillation_gfl import DetectionDistiller_GFL

__all__ = [
    'DetectionDistiller_FR', 'DetectionDistiller_CM', 'DetectionDistiller_Rep',
    'DetectionDistiller_Retina', 'DetectionDistiller_FCOS', 'DetectionDistiller_GFL'
]