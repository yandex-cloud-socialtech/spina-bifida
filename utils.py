import torch
import numpy as np

from ultralytics import YOLO
from monai.networks.nets import densenet121
from monai.transforms import (
    Transform, 
    Compose, 
    EnsureChannelFirst, 
    ScaleIntensity, 
    Resize, 
    EnsureType
)

from monai.visualize import GradCAMpp
from matplotlib import colormaps
from PIL import Image
import io

class PILToNumpy(Transform):
    def __call__(self, pil_image):
        """
        Преобразует изображение PIL в массив NumPy.
        """
        if not isinstance(pil_image, Image.Image):
            raise ValueError("Input must be a PIL Image")

        np_image = np.array(pil_image)
        
        if pil_image.mode == 'L':
            np_image = np.expand_dims(np_image, axis=2)
        
        return np_image.transpose(1, 0, 2)

class MedicalImageProcessor:
    """
    """
    def __init__(self, yolo_model_path, axial_quality_model_path, axial_pathology_model_path, sagittal_quality_model_path, sagittal_pathology_model_path, device):
        self.device = device
        self.yolo_model = YOLO(yolo_model_path).to(device)
        self.axial_quality_model = self._load_model(axial_quality_model_path)
        self.axial_pathology_model = self._load_model(axial_pathology_model_path)
        self.sagittal_quality_model = self._load_model(sagittal_quality_model_path)
        self.sagittal_pathology_model = self._load_model(sagittal_pathology_model_path)
        self.transform = Compose([
                PILToNumpy(),
                EnsureChannelFirst(channel_dim=-1),
                ScaleIntensity(),
                Resize(spatial_size=(255, 255), mode='area'),
                EnsureType(),
            ])
        self.plane_type = {1: 'сагиттальной', 2: 'аксиальной'}

    @staticmethod
    def _crop_image(original_img, roi_bounding_box):
        return original_img.crop(roi_bounding_box)

    def _load_model(self, path):
        model = densenet121(spatial_dims=2, in_channels=3, out_channels=2, pretrained=True)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        
        return model
        
    def object_detection(self, img, conf=0.05):
        predictions = self.yolo_model.predict(img, verbose=False, conf=conf)[0]
        if predictions.boxes.shape[0] == 0:
            return None, None, None  # Возвращаем значения None, если объекты не обнаружены
        boxes = predictions.boxes.data.cpu().detach().numpy()
        x1, y1, x2, y2, conf, plane = boxes[0]
        
        return [x1, y1, x2, y2], conf, plane

    def get_prediction(self, cropped_img_tensor, model):

        outputs = model(cropped_img_tensor)
        prob = torch.sigmoid(outputs[0][1]).item()

        return prob

    def get_heatmap(self, cropped_img_tensor, model, grad_cam, cmap='RdBu', alpha=0.5):
        
        grad_cam_result = grad_cam(x=cropped_img_tensor) 
        cam_image = grad_cam_result[0]
        
        image_np = cropped_img_tensor.squeeze().cpu().numpy()[0]
        cam_image = cam_image.cpu().numpy()[0]
        cam_image = cam_image - cam_image.min()
        cam_image = cam_image / cam_image.max()
        
        cmap = colormaps.get_cmap(cmap)
        cam_colored = cmap(cam_image)  
        
        cam_colored_rgb = (cam_colored[..., :3] * 255).astype(np.uint8)
        cam_image_pil = Image.fromarray(cam_colored_rgb)
        
        image_np_norm = image_np - image_np.min()
        image_np_norm = image_np_norm / image_np_norm.max()
        image_np_rgb = (image_np_norm * 255).astype(np.uint8)
        if image_np_rgb.ndim == 2:
            image_np_rgb = np.stack([image_np_rgb] * 3, axis=-1)
        image_np_pil = Image.fromarray(image_np_rgb, mode='RGB')
        
        blended_image = Image.blend(image_np_pil, cam_image_pil, alpha=0.5)

        return blended_image

    def process_image(self, img_bytes, img_name):
        
        img = Image.open(io.BytesIO(img_bytes))
        boxes, conf, plane = self.object_detection(img)
        if boxes is None:
            return {
                "img_name": img_name,
                "error": "No objects detected in the image."
            }

        cropped_img = self._crop_image(img, boxes)

        if plane == 2:
            quality_model = self.axial_quality_model
            pathology_model = self.axial_pathology_model
        else:
            quality_model = self.sagittal_quality_model
            pathology_model = self.sagittal_pathology_model

        quality_grad_cam = GradCAMpp(nn_module=quality_model, target_layers='class_layers.relu')
        pathology_grad_cam = GradCAMpp(nn_module=pathology_model, target_layers='class_layers.relu')

        cropped_img_tensor = self.transform(cropped_img).unsqueeze(0).to(self.device)
        quality_prob, quality_heatmap = self.get_prediction(cropped_img_tensor, quality_model), self.get_heatmap(cropped_img_tensor, quality_model, quality_grad_cam)
        pathology_prob, pathology_heatmap = self.get_prediction(cropped_img_tensor, pathology_model), self.get_heatmap(cropped_img_tensor, quality_model, pathology_grad_cam)

        result = {
            "img_name": img_name,
            "cropped_img": cropped_img,
            "plane": {"prediction_prob": conf, "type": self.plane_type[plane]},
            "quality": {"prediction_prob": np.round(quality_prob, 2), "heatmap": quality_heatmap.transpose(Image.FLIP_LEFT_RIGHT).rotate(90).resize(cropped_img.size)},
            "pathology": {"prediction_prob": np.round(pathology_prob, 2), "heatmap": pathology_heatmap.transpose(Image.FLIP_LEFT_RIGHT).rotate(90).resize(cropped_img.size)}
        }
        return result