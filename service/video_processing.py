import torch
import torch.nn as nn 
import torchvision.transforms as T
import cv2
import numpy as np 
from PIL import Image, ImageDraw
from models.rtdetrv2_pytorch.src.core import YAMLConfig
from models.sort_tracker.sort import Sort, KalmanBoxTracker


class DetectorInfer:

    def __init__(self, device: str = 'cpu'):
        self.device = device
        checkpoint_path = 'weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'
        cfg = YAMLConfig('models/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml', 
                         resume=checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self, ) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.model = Model().to(self.device)

    
    def predict(self, imgs: list):
        imgs_data = []
        orig_sizes = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            w, h, _ = img.shape
            orig_size = torch.tensor([w, h]).to(self.device)

            transforms = T.Compose([
                T.ToTensor(),
                T.Resize((640, 640)),
            ])
            im_data = transforms(img).to(self.device)
            orig_sizes.append(orig_size)
            imgs_data.append(im_data)

        output = self.model(torch.stack(imgs_data).to(self.device), torch.stack(orig_sizes).to(self.device))
        labels, boxes, scores = output
        return labels, boxes, scores


class FrameProcessor:

    # category_index of person is 0
    def __init__(self, detector: DetectorInfer, category_index: int = 0, **sort_kargs):
        self.category_index = category_index
        self.detector = detector
        self.sort_tracker = Sort(**sort_kargs)
    

    def update(self, frames: list, thr: float = .6):
        labels, boxes, scores = self.detector.predict(frames)
        for i in range(len(frames)):
            im_boxes = boxes[i][torch.logical_and(labels[i] == self.category_index, scores[i] > thr)]
            im_scores = scores[i][torch.logical_and(labels[i] == self.category_index, scores[i] > thr)]
            im_boxes = im_boxes.cpu().detach().numpy()
            im_scores = im_scores.cpu().detach().numpy()[..., None]
            sort_input = np.concatenate((im_boxes, im_scores), axis=-1)
            sort_output = self.sort_tracker.update(sort_input)


    def get_count(self):
        return self.sort_tracker.count_bboxs
