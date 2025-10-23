"""Pose extraction using MediaPipe.

Much faster and easier to use than OpenPose/tf-pose-estimation.
"""

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class PoseExtractor:
    """Extract human pose keypoints using MediaPipe."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe Pose.

        Args:
            static_image_mode: If True, treats each image independently
            model_complexity: 0=Lite, 1=Full, 2=Heavy
            enable_segmentation: Generate segmentation mask
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # MediaPipe returns 33 landmarks
        self.num_keypoints = 33

    def extract_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose keypoints from image.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Array of shape (33, 3) containing [x, y, visibility] for each keypoint,
            or None if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process
        results = self.pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks])

        return keypoints

    def extract_normalized_keypoints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract keypoints normalized relative to hip center. This makes the
        representation translation and scale invariant.

        Returns:
            Array of shape (33, 2) with normalized x, y coordinates
        """
        keypoints = self.extract_keypoints(image)

        if keypoints is None:
            return None

        # Get hip center (average of left and right hip)
        # Left hip = 23, Right hip = 24 in MediaPipe
        left_hip = keypoints[23, :2]
        right_hip = keypoints[24, :2]
        hip_center = (left_hip + right_hip) / 2

        # Normalize: subtract hip center
        normalized = keypoints[:, :2] - hip_center

        # Scale normalization: use shoulder-hip distance
        left_shoulder = keypoints[11, :2]
        right_shoulder = keypoints[12, :2]
        shoulder_center = (left_shoulder + right_shoulder) / 2

        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length > 0:
            normalized = normalized / torso_length

        return normalized

    def draw_pose(
        self, image: np.ndarray, keypoints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Draw pose skeleton on image.

        Args:
            image: BGR image
            keypoints: If None, will extract from image

        Returns:
            Image with pose drawn
        """
        if keypoints is None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                annotated = image.copy()
                self.mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                )
                return annotated

        return image

    def __del__(self):
        """Cleanup."""
        if hasattr(self, "pose"):
            self.pose.close()


class PoseClassifier:
    """Simple rule-based pose classifier (sitting/standing/lying)."""

    @staticmethod
    def classify_pose(keypoints: np.ndarray) -> str:
        """Classify pose as sitting, standing, or lying.

        Args:
            keypoints: Array of shape (33, 3) from MediaPipe

        Returns:
            "standing", "sitting", or "lying"
        """
        # Key landmarks
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_knee = keypoints[25]
        right_knee = keypoints[26]
        left_ankle = keypoints[27]
        right_ankle = keypoints[28]
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]

        hip_center = (left_hip[:2] + right_hip[:2]) / 2
        shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2

        # Vertical alignment (hip to shoulder)
        torso_vertical = abs(shoulder_center[1] - hip_center[1])
        torso_horizontal = abs(shoulder_center[0] - hip_center[0])

        # Lying down: torso more horizontal than vertical
        if torso_horizontal > torso_vertical * 1.5:
            return "lying"

        # Sitting vs standing: check knee-hip-ankle angles
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        avg_knee_y = (left_knee[1] + right_knee[1]) / 2
        avg_ankle_y = (left_ankle[1] + right_ankle[1]) / 2

        # If knees are significantly bent (knee below hip, ankle below knee)
        # and the knee-ankle distance is small, likely sitting
        hip_knee_dist = avg_knee_y - avg_hip_y
        knee_ankle_dist = avg_ankle_y - avg_knee_y

        if hip_knee_dist > 0.1 and knee_ankle_dist < 0.15:
            return "sitting"

        return "standing"
