"""Main predictor class for easy inference."""

from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from hac.common.class_names import STANFORD40_CLASSES, resolve_class_names
from hac.common.transforms import get_inference_transforms
from hac.image.inference.pose_extractor import PoseClassifier, PoseExtractor
from hac.image.models.classifier import ActionClassifier


class ActionPredictor:
    """High-level API for action prediction.

    Handles the full pipeline: pose extraction → classification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_pose_estimation: bool = True,
        pose_model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model weights
            class_names: List of class names (in order)
            device: "cuda" or "cpu"
            use_pose_estimation: Whether to use MediaPipe pose estimation
            pose_model_path: Optional separate model for pose classification
            model_name: timm architecture, used when the weights carry no config
                (e.g. ``.safetensors`` files). Defaults to ``resnet34``.
            num_classes: Class count, used when the weights carry no config.
                Defaults to 40 (Stanford40).
        """
        self.device = torch.device(device)
        self._fallback_model_name = model_name
        self._fallback_num_classes = num_classes
        self.use_pose_estimation = use_pose_estimation

        # Initialize pose extractor if needed
        if use_pose_estimation:
            self.pose_extractor = PoseExtractor(
                static_image_mode=True, model_complexity=1, min_detection_confidence=0.5
            )
            self.pose_classifier = PoseClassifier()

        # Load model (keeps the raw checkpoint so stored class names can be used)
        self._checkpoint = None
        if model_path:
            self.model = self._load_model(model_path)
        else:
            # Default to pretrained on ImageNet (for demo purposes)
            print("Warning: No model_path provided, using untrained model")
            self.model = ActionClassifier(
                model_name="mobilenetv3_small_100", num_classes=40, pretrained=True
            )

        self.model = self.model.to(self.device)
        self.model.eval()

        # Class names: explicit argument > checkpoint > Stanford40 default
        num_classes = self.model.num_classes
        if class_names:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Got {len(class_names)} class names for a "
                    f"{num_classes}-class model"
                )
            self.class_names = list(class_names)
        else:
            self.class_names = resolve_class_names(
                self._checkpoint, num_classes, STANFORD40_CLASSES
            )

        # Get transforms
        self.transform = get_inference_transforms()

    def _load_model(self, model_path: str):
        """Load model from local path or HuggingFace Hub."""

        # Handle HuggingFace URLs
        if model_path.startswith(("https://huggingface.co/", "hf://")):
            print("Downloading model from HuggingFace...")

            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models. "
                    "Install with: pip install huggingface_hub"
                )

            # Parse HuggingFace URL
            if model_path.startswith("https://huggingface.co/"):
                parts = model_path.replace("https://huggingface.co/", "").split("/")
                repo_id = f"{parts[0]}/{parts[1]}"

                if "resolve" in parts:
                    resolve_idx = parts.index("resolve")
                    filename = "/".join(parts[resolve_idx + 2 :])
                else:
                    filename = "model.safetensors"

            elif model_path.startswith("hf://"):
                path = model_path.replace("hf://", "")
                parts = path.split("/")
                repo_id = f"{parts[0]}/{parts[1]}"
                filename = (
                    "/".join(parts[2:]) if len(parts) > 2 else "model.safetensors"
                )

            # Download from HuggingFace Hub
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, cache_dir=None
                )
                print(f"✓ Downloaded to: {model_path}")
            except Exception as e:
                raise ValueError(
                    f"Failed to download model from HuggingFace: {e}\n"
                    f"Repo: {repo_id}\n"
                    f"File: {filename}"
                )

        # Check if it's a SafeTensors file
        if model_path.endswith(".safetensors"):
            # Load SafeTensors format
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise ImportError(
                    "safetensors is required to load .safetensors files. "
                    "Install with: pip install safetensors"
                )

            state_dict = load_file(model_path)
            config = {}  # SafeTensors doesn't store config

        else:
            # Load PyTorch checkpoint - FIX: Add weights_only=False for compatibility
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # Extract state dict and config
            if isinstance(checkpoint, dict):
                self._checkpoint = {"class_names": checkpoint.get("class_names")}
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                config = checkpoint.get("config", {})
            else:
                state_dict = checkpoint
                config = {}

        # Architecture: checkpoint config > constructor arguments > defaults
        model_name = config.get("model_name") or self._fallback_model_name
        num_classes = config.get("num_classes") or self._fallback_num_classes
        if model_name is None:
            model_name = "resnet34"
            print(
                "Warning: weights carry no architecture info; assuming "
                f"'{model_name}'. Pass model_name=... (CLI: --model_name) if wrong."
            )
        if num_classes is None:
            num_classes = 40

        # Create model
        from hac.image.models.classifier import create_model

        model = create_model(
            model_type="action",
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )

        # Load weights
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                f"Weights at {model_path} do not match a '{model_name}' model with "
                f"{num_classes} classes. If the file has no embedded config (e.g. "
                "a .safetensors export), pass the correct model_name/num_classes "
                "(CLI: --model_name / --num_classes)."
            ) from e
        model = model.to(self.device)
        model.eval()

        print(f"✓ Model loaded: {model_name} ({num_classes} classes)")

        return model

    @torch.no_grad()
    def predict_image(
        self, image: Union[str, np.ndarray], return_pose: bool = True, top_k: int = 5
    ) -> Dict:
        """Predict action from a single image.

        Args:
            image: Path to image or numpy array (BGR)
            return_pose: Also return pose classification
            top_k: Number of top predictions to return

        Returns:
            Dictionary with predictions
        """
        # Load image if path
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

        results = {}

        # Pose estimation
        if self.use_pose_estimation and return_pose:
            keypoints = self.pose_extractor.extract_keypoints(image)
            if keypoints is not None:
                pose_class = self.pose_classifier.classify_pose(keypoints)
                results["pose"] = {
                    "class": pose_class,
                    "keypoints": keypoints,
                }
                # Draw pose on image for visualization
                results["pose_image"] = self.pose_extractor.draw_pose(image.copy())

        # Action classification
        # Preprocess image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(
            probs[0], k=min(top_k, len(self.class_names))
        )

        predictions = [
            {"class": self.class_names[idx], "confidence": prob.item()}
            for prob, idx in zip(top_probs, top_indices)
        ]

        results["action"] = {
            "predictions": predictions,
            "top_class": predictions[0]["class"],
            "top_confidence": predictions[0]["confidence"],
        }

        return results

    @torch.no_grad()
    def predict_video(
        self, video_path: str, sample_rate: int = 5, aggregate_method: str = "voting"
    ) -> Dict:
        """Predict action from video by sampling frames.

        Args:
            video_path: Path to video file
            sample_rate: Sample every N frames
            aggregate_method: "voting" or "average"

        Returns:
            Dictionary with aggregated predictions
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        frame_predictions = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_count % sample_rate == 0:
                try:
                    result = self.predict_image(frame, return_pose=False)
                    frame_predictions.append(result["action"]["top_class"])
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")

            frame_count += 1

        cap.release()

        if not frame_predictions:
            raise ValueError(
                f"No frames could be classified from {video_path} "
                f"({frame_count} frames read)"
            )

        # Aggregate predictions
        if aggregate_method == "voting":
            # Most common prediction
            from collections import Counter

            vote_counts = Counter(frame_predictions)
            final_prediction = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[final_prediction] / len(frame_predictions)
        else:
            # This would require storing all probabilities, simplified here
            final_prediction = max(set(frame_predictions), key=frame_predictions.count)
            confidence = frame_predictions.count(final_prediction) / len(
                frame_predictions
            )

        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "sampled_frames": len(frame_predictions),
            "prediction": final_prediction,
            "confidence": confidence,
            "frame_predictions": frame_predictions,
        }

    def predict_webcam(self, camera_id: int = 0):
        """Run real-time prediction on webcam. Press 'q' to quit.

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")

        print("Starting webcam prediction. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Predict
            result = self.predict_image(frame, return_pose=True)

            # Annotate frame
            display_frame = frame.copy()

            # Draw pose if available
            if "pose_image" in result:
                display_frame = result["pose_image"]

            # Add text annotations
            y_offset = 30

            if "pose" in result:
                pose_text = f"Pose: {result['pose']['class']}"
                cv2.putText(
                    display_frame,
                    pose_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += 30

            if "action" in result:
                action = result["action"]["predictions"][0]
                action_text = f"Action: {action['class']} ({action['confidence']:.2f})"
                cv2.putText(
                    display_frame,
                    action_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Display
            cv2.imshow("Human Action Classification", display_frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
