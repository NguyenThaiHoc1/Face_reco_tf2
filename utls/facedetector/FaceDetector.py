import numpy as np
from mtcnn import MTCNN

from utls.facealigner.FaceAligner import FaceAligner
from utls.other import other_face as utils


class FaceDetector:
    def __init__(self, image_size=160, margin=10, detect_multiple_faces=False, min_face_size=20):
        self.aligner = FaceAligner(desiredFaceWidth=image_size, margin=margin)
        self.detector = MTCNN(min_face_size=min_face_size)
        self.detect_multiple_faces = detect_multiple_faces

    def extract(self, img):
        """ Extract the portions of a single img or frame including faces """
        bounding_box, landmarks = self.detect(img)
        return [self.aligner.align(img, det) for det in zip(bounding_box, landmarks)]

    def detect(self, img):
        bounding_boxes = self.detector.detect_faces(img)
        nrof_faces = len(bounding_boxes)
        if nrof_faces <= 0:
            return [], []

        det = np.array([utils.fix_box(b['box']) for b in bounding_boxes])
        img_size = np.asarray(img.shape)[0:2]

        if nrof_faces > 1 and not self.detect_multiple_faces:
            # select the biggest and most central
            bounding_box_size = det[:, 2] * det[:, 3]
            img_center = img_size / 2
            offsets = np.vstack([det[:, 0] + (det[:, 2] / 2) - img_center[1],
                                 det[:, 1] + (det[:, 3] / 2) - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # some extra weight on the centering
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
            det_arr = [det[index, :]]
            landmarks = [bounding_boxes[index]['keypoints']]
        else:
            det_arr = [np.squeeze(d) for d in det]
            landmarks = [b['keypoints'] for b in bounding_boxes]

        return det_arr, landmarks
