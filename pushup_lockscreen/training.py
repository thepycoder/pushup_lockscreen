import joblib
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from clearml import Task
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from preprocessing import imgaug_to_numpy, select_landmarks
from augmentation import LandMarkAugmentation
import global_config

task = Task.init(project_name='pushup_lockscreen', task_name='training', reuse_last_task_id=False)


class ModelTrainer:
    def __init__(self):
        self.params = {
            'test_size': 0.2,
            'seed': 42,
            'estimator': 'SVC',
            'SVC': {
                'C': 1,
                'kernel': 'linear',
                'gamma': 0.001
            },
            'RF': {
                'n_estimators': 100,
                'max_depth': 2,
                'max_features': 1
            },
            'ADA': {
                'n_estimators': 50,
                'learning_rate': 1.0
            },
            'MLP': {
                'hidden_layer_sizes': (50, 50),
                'learning_rate_init': 0.001
            }
        }
        self.selected_keypoints = global_config.SELECTED_KEYPOINTS
        # Get the path to the landmarks, expect only csv files
        self.landmark_path = '../landmarks'
        # Get the path to the raw_data, expect only jpg files
        self.raw_data_path = '../raw_data'
        # The amount of time the whole batch will be augmented and added to the original
        self.augmentation_ratio = 10
        self.enable_debug_images = True
        task.connect(self)

        self.augmenter = LandMarkAugmentation(self.params['seed'])

    def get_landmarks(self):
        landmarks = []
        landmark_labels = []

        images = []
        image_labels = []

        for csv_name in sorted(glob.glob(os.path.join(self.landmark_path, '**/*.csv'))):
            landmarks.append(np.loadtxt(csv_name, delimiter=','))
            landmark_labels.append(os.path.basename(os.path.dirname(csv_name)))

        for image_name in sorted(glob.glob(os.path.join(self.raw_data_path, '**/*.jpg'))):
            images.append(cv2.imread(image_name))
            image_labels.append(os.path.basename(os.path.dirname(image_name)))

        return np.array(landmarks), np.array(landmark_labels), np.array(images), np.array(image_labels)

    def augment(self, images, X, y, train_test):
        X_augmented = X.copy()
        y_augmented = y.copy()
        augmented_landmarks = []
        augmented_images = []
        for _ in range(self.augmentation_ratio):
            augmented_images, augmented_landmarks = \
                self.augmenter.get_augmented_batch(images,
                                                   X.reshape(-1, len(self.selected_keypoints), 2)
                                                   )
            np.append(images, augmented_images, axis=0)
            np_landmarks = imgaug_to_numpy(augmented_landmarks, X.shape[1])
            X_augmented = np.append(X_augmented, np_landmarks, 0)
            y_augmented = np.append(y_augmented, y)
        if self.enable_debug_images:
            keypoint_image = augmented_landmarks[0].draw_on_image(augmented_images[0], size=7)
            task.get_logger().report_image('Augmentation Debug Images', f'{train_test} - {y[0]}',
                                                image=keypoint_image[..., ::-1])
        return X_augmented, y_augmented

    def preprocess_landmarks(self, selected_landmarks, landmark_labels, images, image_labels):
        X_train, X_test, y_train, y_test, images_train, images_test, image_labels_train, image_labels_test = \
            train_test_split(selected_landmarks, landmark_labels, images, image_labels,
                             test_size=self.params['test_size'], random_state=self.params['seed'])

        X_train_augmented, y_train_augmented = self.augment(images_train, X_train, y_train, 'training')
        X_test_augmented, y_test_augmented = self.augment(images_test, X_test, y_test, 'validation')

        print(f'Training: {X_train_augmented.shape}')
        print(f'Testing: {X_test_augmented.shape}')

        # Normalize the landmarks of course
        scaler = StandardScaler()
        # Only fit the scaler on training data! Otherwise knowledge of test data will leak into training set
        scaler.fit(X_train_augmented)
        joblib.dump(scaler, 'scaler.pkl', compress=True)
        task.upload_artifact(name='scaler_remote', artifact_object='scaler.pkl')
        X_train_augmented = scaler.transform(X_train_augmented)
        X_test_augmented = scaler.transform(X_test_augmented)

        le = LabelEncoder()
        le.fit(y_train_augmented)
        y_train_augmented = le.transform(y_train_augmented)
        y_test_augmented = le.transform(y_test_augmented)

        return X_train_augmented, y_train_augmented, X_test_augmented, y_test_augmented

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test):
        if self.params['estimator'] == 'SVC':
            model = SVC(**self.params['SVC'], random_state=self.params['seed'])
        elif self.params['estimator'] == 'RF':
            model = RandomForestClassifier(**self.params['RF'], random_state=self.params['seed'])
        elif self.params['estimator'] == 'ADA':
            model = AdaBoostClassifier(**self.params['ADA'], random_state=self.params['seed'])
        elif self.params['estimator'] == 'MLP':
            model = MLPClassifier(**self.params['MLP'], random_state=self.params['seed'])
        else:
            print("Classifier type not implemented")
            return

        model.fit(X_train, y_train)
        task.get_logger().report_scalar(title=self.params['estimator'],
                                        series='ROC AUC Test',
                                        value=roc_auc_score(y_test, model.predict(X_test)),
                                        iteration=0)
        task.get_logger().report_scalar(title=self.params['estimator'],
                                        series='ROC AUC Train',
                                        value=roc_auc_score(y_train, model.predict(X_train)),
                                        iteration=0)
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title('Confusion Matrix')
        plt.show()
        # Save the model file
        joblib.dump(model, 'model.pkl', compress=True)
        # task.upload_artifact(name='model_remote', artifact_object='model.pkl')

    def run(self):
        # Landmarks and images will be stored in the instance itself for potential later retrieval
        landmarks, landmark_labels, images, image_labels = self.get_landmarks()
        selected_landmarks = select_landmarks(landmarks, self.selected_keypoints)
        X_train, y_train, X_test, y_test = \
            self.preprocess_landmarks(selected_landmarks, landmark_labels, images, image_labels)
        self.train_and_evaluate_model(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
