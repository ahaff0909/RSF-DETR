# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector. RT-DETR offers real-time
performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT. It features an efficient
hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

For more information on RT-DETR, visit: https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RSFDETR(Model):
    """Training, validation and inference interface for RSF-DETR.

    Attributes:
        model (str): Model YAML or PyTorch checkpoint.
    """

    def __init__(self, model='configs/rsf-detr.yaml') -> None:
        """
        Initializes the RT-DETR model with the given pre-trained model file. Supports .pt and .yaml formats.

        Args:
            model (str): Path to an RSF-DETR YAML or PyTorch checkpoint.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt, *.yaml, or *.yml files.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            dict: A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}
