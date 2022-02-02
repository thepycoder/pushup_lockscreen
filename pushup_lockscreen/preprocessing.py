import numpy as np


def imgaug_to_numpy(landmark_list, np_shape):
    np_landmarks = []
    for list_of_keypoints in landmark_list:
        list_of_np_keypoints = []
        for keypoint in list_of_keypoints:
            np_keypoint = np.array([keypoint.x, keypoint.y])
            list_of_np_keypoints.append(np_keypoint)
        np_landmarks.append(list_of_np_keypoints)
    np_landmarks = np.array(np_landmarks).reshape(-1, np_shape)
    return np_landmarks


def select_landmarks(landmarks, keypoints):
    # Keep only keypoints in the list; resulting shape: (-1, 14, 3). 3d coords for each selected keypoint
    selected_landmarks = np.take(landmarks, keypoints, axis=1)
    # Remove the 3rd dim coordinate and flatten the x and y coords into 1 vector
    selected_landmarks = np.array(selected_landmarks)[:, :, :2]
    selected_landmarks = selected_landmarks.reshape(selected_landmarks.shape[0], -1)
    return selected_landmarks
