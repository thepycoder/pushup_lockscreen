import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2
import numpy as np


class LandMarkAugmentation:
    def __init__(self, seed=42):
        ia.seed(seed)
        self.imgaug_pipeline = self.create_imgaug()
        self.max_jitter_multiplier = 5
        self.jitter_coef = 10 / self.max_jitter_multiplier  # absolute maximum deviation in pixels

    def get_jitter_normal(self):
        jitter = np.random.normal()
        if jitter > self.max_jitter_multiplier:
            jitter = self.max_jitter_multiplier
        elif jitter < -self.max_jitter_multiplier:
            jitter = -self.max_jitter_multiplier
        return jitter

    def create_imgaug(self):
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Affine(
                rotate=(-5, 5),
                scale=(0.7, 1),
                shear=(-5, 5),
                translate_percent=(0, 0.05)
            )
        ])
        return seq

    def get_augmented_batch(self, image_list, landmark_list):
        original_keypoints = []
        jittered_keypoints = []

        for image, landmarks_array in zip(image_list, landmark_list):
            original_keypoints_instance = []
            jittered_keypoints_instance = []
            for point in landmarks_array:
                # Then augment using jitter
                original_keypoints_instance.append(
                    Keypoint(x=point[0],
                             y=point[1])
                )
                jittered_keypoints_instance.append(
                    Keypoint(x=point[0] + self.get_jitter_normal() * self.jitter_coef,
                             y=point[1] + self.get_jitter_normal() * self.jitter_coef)
                )
            original_kpsoi = KeypointsOnImage(original_keypoints_instance, shape=image.shape)
            jittered_kpsoi = KeypointsOnImage(jittered_keypoints_instance, shape=image.shape)
            # keypoint_image = kps.draw_on_image(image, size=7)
            # cv2.imwrite('test_keypoints.jpg', keypoint_image)
            # break
            original_keypoints.append(original_kpsoi)
            jittered_keypoints.append(jittered_kpsoi)

        # Now run the actualy augmentation
        image_list_aug, keypoints_aug = self.imgaug_pipeline(images=image_list, keypoints=jittered_keypoints)

        return image_list_aug, keypoints_aug

        # for i, (image_aug, kpsoi_aug) in enumerate(zip(image_list_aug, keypoints_aug)):
        #     result = np.hstack([
        #         original_keypoints[i].draw_on_image(image_list[i], size=7),
        #         kpsoi_aug.draw_on_image(image_aug, size=7)
        #     ])
        #     cv2.imwrite(f'augmented_images/test_keypoints_{i}.jpg', result)


augmenter = LandMarkAugmentation()
augmented_images, augmented_landmarks = augmenter.get_augmented_batch([np.zeros((300, 300, 3))]*20, np.zeros((20, 14, 2)))