import argparse
import logging
import os
import sys
import time
import tqdm
import cv2

from utls.facedetector.FaceDetector import FaceDetector
from utls.other import other_face as utils

logger = logging.getLogger('crawler')


def main(project='general', discard_multi_face=False):
    image_dir = os.path.expanduser(os.path.join('data/training_img/', project))

    al_image_dir = os.path.expanduser(os.path.join('data/training_img_aligned/', project))
    os.makedirs(al_image_dir, exist_ok=True)

    start = time.time()
    detector = FaceDetector(detect_multiple_faces=True, image_size=112)  # resize image

    for folder_name in tqdm.tqdm(sorted(os.listdir(image_dir))):
        dst_path_abs = os.path.join(al_image_dir, folder_name)
        src_path_abs = os.path.join(image_dir, folder_name)
        if not os.path.exists(dst_path_abs):
            os.makedirs(dst_path_abs, exist_ok=True)

        for f in sorted(os.listdir(src_path_abs)):
            filename = f.rsplit('.', 1)[0]
            image = utils.load_gray(os.path.join(src_path_abs, f))
            extracted_faces = detector.extract(image)
            if discard_multi_face and len(extracted_faces) > 1:
                continue
            for i, face in enumerate(extracted_faces):
                output_filename = os.path.join(dst_path_abs, '%s_%d.jpg' % (filename, i))
                cv2.imwrite(output_filename, face)

    end = time.time()
    logger.info("Time elapsed: %.2f seconds", end - start)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default='lfw',
                        help='Name of the collection to be part of')
    parser.add_argument("--discard_multi_faces", default=False, action='store_true',
                        help='If set, discard the pictures that contain more that one face.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args.project, args.discard_multi_faces)
