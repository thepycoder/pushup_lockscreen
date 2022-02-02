import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
if '.' not in os.sys.path:
    os.sys.path.append('.')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, confusion_matrix

from pushup_lockscreen.augmentation import LandMarkAugmentation


from clearml import Task
task = Task.init(project_name='HPO Blogpost Simple', task_name='base_task', reuse_last_task_id=False)


def get_data():
    landmarks = []
    landmark_labels = []

    images = []
    image_labels = []

    for csv_name in sorted(glob.glob('landmarks/**/*.csv')):
        landmarks.append(np.loadtxt(csv_name, delimiter=','))
        landmark_labels.append(os.path.basename(os.path.dirname(csv_name)))

    for image_name in sorted(glob.glob('raw_data/**/*.jpg')):
        images.append(cv2.imread(image_name))
        image_labels.append(os.path.basename(os.path.dirname(image_name)))

    print(f'Loaded {len(landmarks)} samples')

    landmarks = np.array(landmarks)
    landmark_labels = np.array(landmark_labels)
    images = np.array(images)
    image_labels = np.array(image_labels)

    return landmarks, landmark_labels, images, image_labels


def filter_landmarks(params, landmarks):
    # Keep only keypoints in the list; resulting shape: (-1, <selected>, 3). 3d coords for each selected keypoint
    selected_landmarks = np.take(landmarks, params['selected_keypoints'], axis=1)
    # Remove the 3rd dim coordinate and flatten the x and y coords into 1 vector
    selected_landmarks = np.array(selected_landmarks)[:, :, :2]
    selected_landmarks = selected_landmarks.reshape(selected_landmarks.shape[0], -1)

    return selected_landmarks


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


def augment_data(params, X_train, y_train, X_test, y_test, images_train, images_test):
    augmenter = LandMarkAugmentation(params['seed'])
    X_train_augmented = X_train.copy()
    X_test_augmented = X_test.copy()
    y_train_augmented = y_train.copy()
    y_test_augmented = y_test.copy()

    # add 10x the amount of original data in augmented versions
    for _ in range(params['augment_factor']):
        augmented_training_images, augmented_training_landmarks = \
            augmenter.get_augmented_batch(images_train, X_train.reshape(-1, len(params['selected_keypoints']), 2))
        log_augmentation_samples(augmented_training_landmarks[0], augmented_training_images[0])

        np.append(images_train, augmented_training_images, axis=0)
        np_landmarks = imgaug_to_numpy(augmented_training_landmarks, X_train.shape[1])
        X_train_augmented = np.append(X_train_augmented, np_landmarks, 0)
        y_train_augmented = np.append(y_train_augmented, y_train)

    for _ in range(params['augment_factor']):
        augmented_testing_images, augmented_testing_landmarks = \
            augmenter.get_augmented_batch(images_test, X_test.reshape(-1, len(params['selected_keypoints']), 2))
        np.append(images_test, augmented_testing_images, axis=0)
        np_landmarks = imgaug_to_numpy(augmented_testing_landmarks, X_test.shape[1])
        X_test_augmented = np.append(X_test_augmented, np_landmarks, 0)
        y_test_augmented = np.append(y_test_augmented, y_test)

    print(f'Training: {X_train_augmented.shape}')
    print(f'Testing: {X_test_augmented.shape}')

    return X_train_augmented, y_train_augmented, X_test_augmented, y_test_augmented, images_train, images_test


def log_augmentation_samples(landmark, image):
    keypoint_image = landmark.draw_on_image(image, size=7)
    task.get_logger().report_image(title='Augmentation', series='Augmentation Sample', image=keypoint_image)


def scale_data(X_train, y_train, X_test, y_test):
    label_encoder = LabelEncoder()
    data_scaler = StandardScaler()
    # Only fit the scaler on training data! Otherwise knowledge of test data will leak into training set
    data_scaler.fit(X_train)
    X_train = data_scaler.transform(X_train)
    X_test = data_scaler.transform(X_test)

    # Same for labels, they have to be numbers to be able to use all sklearn classifiers
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test


def train_model(params, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(**params['RF'], random_state=params['seed'])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    task.get_logger().report_scalar(title='Random Forest',
                                    series='ROC AUC Test',
                                    value=roc_auc_score(y_test, y_pred),
                                    iteration=0)
    task.get_logger().report_scalar('Random Forest', 'ROC AUC Train',
                                    roc_auc_score(y_train, model.predict(X_train)), 0)
    cm = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    task.get_logger().report_matplotlib_figure('Confusion Matrix Sklearn', 'RF', cm.figure_)


def run(params):
    print(params)
    landmarks, landmark_labels, images, image_labels = get_data()
    landmarks = filter_landmarks(params, landmarks)
    X_train, X_test, y_train, y_test, images_train, images_test, image_labels_train, image_labels_test = \
        train_test_split(landmarks, landmark_labels, images, image_labels,
                         test_size=params['test_size'], random_state=params['seed'])

    X_train, y_train, X_test, y_test, images_train, images_test = \
        augment_data(params, X_train, y_train, X_test, y_test, images_train, images_test)

    train_model(params, X_train, y_train, X_test, y_test)



    # X_train, y_train, X_test, y_test = \
    #     scale_data(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    params = {
        'selected_keypoints': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               25, 26, 27, 28, 29, 30, 31, 32, 33],
        'test_size': 0.3,
        'augment_factor': 10,
        'seed': 42,
        'RF': {
            'n_estimators': 100,
            'max_depth': 100,
            'max_features': 10
        },
    }
    task.connect(params)
    run(params)
