import os
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    import pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    logger.warning("PyOpenPose not available")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Install with: pip install mediapipe")


class KeypointExtractor:
    """
    비디오에서 키포인트 추출 클래스
    OpenPose 기본, MediaPipe 대안 지원
    """
    
    def __init__(self, method: str = "openpose", confidence_threshold: float = 0.5, config=None):
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold
        self.config = config
        
        if self.method == "openpose" and OPENPOSE_AVAILABLE:
            self._init_openpose()
        elif self.method == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        else:
            raise ValueError(f"Method {method} not available or not installed")
    
    def _init_openpose(self):
        """OpenPose 초기화 - 안정적인 왼손 검출 설정"""
        params = dict()
        
        # Get OpenPose model path from config or use default
        if self.config and hasattr(self.config, 'keypoint_extraction') and hasattr(self.config.keypoint_extraction, 'openpose_model_path'):
            model_path = self.config.keypoint_extraction.openpose_model_path
        else:
            # Fallback to default path
            model_path = "/Users/user/Desktop/openpose/models"
            logger.warning(f"Config not provided, using default OpenPose model path: {model_path}")
        
        params["model_folder"] = model_path
        params["face"] = True
        params["hand"] = True
        params["model_pose"] = "BODY_25"  # 명시적 모델 지정
        self.openpose = op.WrapperPython()
        self.openpose.configure(params)
        self.openpose.start()
        logger.info(f"OpenPose initialized with conservative hand detection settings")
    
    def _init_mediapipe(self):
        """MediaPipe 초기화"""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold
        )
        logger.info("MediaPipe initialized")
    
    def extract_from_video(self, video_path: str, output_dir: str) -> bool:
        """
        비디오에서 키포인트 추출
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 키포인트 JSON 파일들이 저장될 디렉토리
            
        Returns:
            성공 여부
        """
        # 절대 경로로 변환
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Extracting keypoints from {video_path} ({total_frames} frames)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 키포인트 추출
                if self.method == "openpose":
                    keypoints_data = self._extract_openpose(frame)
                elif self.method == "mediapipe":
                    keypoints_data = self._extract_mediapipe(frame)
                else:
                    logger.error(f"Unknown method: {self.method}")
                    return False
                
                # JSON 파일로 저장
                output_file = os.path.join(output_dir, f"frame_{frame_count:06d}_keypoints.json")
                with open(output_file, 'w') as f:
                    json.dump(keypoints_data, f, indent=2)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
        
        logger.info(f"Extraction completed: {frame_count} frames processed")
        return frame_count > 0
    
    def _extract_openpose(self, frame: np.ndarray) -> Dict:
        """OpenPose를 사용한 키포인트 추출"""
        # OpenPose 처리
        datum = op.Datum()
        datum.cvInputData = frame
        self.openpose.emplaceAndPop(op.VectorDatum([datum]))
        
        # 결과 변환 (기존 데이터셋 구조에 맞춤)
        keypoints_data = {
            "version": 1.3,
            "people": {},
            "camparam": {
                "Intrinsics": {"data": ""},
                "CameraMatrix": {"data": ""},
                "Distortion": {"rows": "", "data": ""}
            }
        }
        
        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            person_data = {
                "person_id": -1,  # 1명인 경우 -1로 설정
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            
            # Pose keypoints (25개 * 3차원 -> 75개 값, confidence 포함)
            pose_kp = datum.poseKeypoints[0]  # 첫 번째 사람
            person_data["pose_keypoints_2d"] = pose_kp.flatten().tolist()
            
            # Face keypoints (70개 * 3차원 -> 210개 값, confidence 포함)
            if datum.faceKeypoints is not None and len(datum.faceKeypoints) > 0:
                face_kp = datum.faceKeypoints[0]
                person_data["face_keypoints_2d"] = face_kp.flatten().tolist()
            else:
                person_data["face_keypoints_2d"] = [0.0] * 210  # 70 * 3
            
            # Hand keypoints
            if datum.handKeypoints is not None and len(datum.handKeypoints) > 0:
                if len(datum.handKeypoints[0]) > 0:  # Left hand (21개 * 3차원 -> 63개 값)
                    left_hand_kp = datum.handKeypoints[0][0]
                    person_data["hand_left_keypoints_2d"] = left_hand_kp.flatten().tolist()
                else:
                    person_data["hand_left_keypoints_2d"] = [0.0] * 63  # 21 * 3
                
                if len(datum.handKeypoints[1]) > 0:  # Right hand (21개 * 3차원 -> 63개 값)
                    right_hand_kp = datum.handKeypoints[1][0]
                    person_data["hand_right_keypoints_2d"] = right_hand_kp.flatten().tolist()
                else:
                    person_data["hand_right_keypoints_2d"] = [0.0] * 63  # 21 * 3
            else:
                person_data["hand_left_keypoints_2d"] = [0.0] * 63  # 21 * 3
                person_data["hand_right_keypoints_2d"] = [0.0] * 63  # 21 * 3
            
            keypoints_data["people"] = person_data
        
        return keypoints_data
    
    def _extract_mediapipe(self, frame: np.ndarray) -> Dict:
        """MediaPipe를 사용한 키포인트 추출 (OpenPose 형식으로 변환)"""
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 키포인트 추출
        results = self.holistic.process(rgb_frame)
        
        # 기존 데이터셋 구조로 변환
        keypoints_data = {
            "version": 1.3,
            "people": {},
            "camparam": {
                "Intrinsics": {"data": ""},
                "CameraMatrix": {"data": ""},
                "Distortion": {"rows": "", "data": ""}
            }
        }
        
        if (results.pose_landmarks or results.face_landmarks or 
            results.left_hand_landmarks or results.right_hand_landmarks):
            
            person_data = {
                "person_id": -1,  # 1명인 경우 -1로 설정
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }
            
            # Pose keypoints (25개 * 3차원 -> 75개 값, confidence 포함)
            if results.pose_landmarks:
                pose_keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    pose_keypoints.extend([landmark.x * frame.shape[1], 
                                         landmark.y * frame.shape[0],
                                         landmark.visibility])
                # MediaPipe 33개 -> OpenPose 25개 매핑
                pose_keypoints_25 = self._mediapipe_to_openpose_pose(pose_keypoints)
                person_data["pose_keypoints_2d"] = pose_keypoints_25
            else:
                person_data["pose_keypoints_2d"] = [0.0] * 75  # 25 * 3
            
            # Face keypoints (70개 * 3차원 -> 210개 값, confidence 포함)
            if results.face_landmarks:
                face_keypoints = []
                # MediaPipe face landmarks 468개 중 70개 선택
                selected_indices = self._get_face_landmark_indices()
                for i in selected_indices:
                    landmark = results.face_landmarks.landmark[i]
                    face_keypoints.extend([landmark.x * frame.shape[1], 
                                         landmark.y * frame.shape[0],
                                         landmark.visibility if hasattr(landmark, 'visibility') else 1.0])
                person_data["face_keypoints_2d"] = face_keypoints
            else:
                person_data["face_keypoints_2d"] = [0.0] * 210  # 70 * 3
            
            # Left hand keypoints (21개 * 3차원 -> 63개 값, confidence 포함)
            if results.left_hand_landmarks:
                left_hand_keypoints = []
                for landmark in results.left_hand_landmarks.landmark:
                    left_hand_keypoints.extend([landmark.x * frame.shape[1], 
                                              landmark.y * frame.shape[0],
                                              landmark.visibility if hasattr(landmark, 'visibility') else 1.0])
                person_data["hand_left_keypoints_2d"] = left_hand_keypoints
            else:
                person_data["hand_left_keypoints_2d"] = [0.0] * 63  # 21 * 3
            
            # Right hand keypoints (21개 * 3차원 -> 63개 값, confidence 포함)
            if results.right_hand_landmarks:
                right_hand_keypoints = []
                for landmark in results.right_hand_landmarks.landmark:
                    right_hand_keypoints.extend([landmark.x * frame.shape[1], 
                                               landmark.y * frame.shape[0],
                                               landmark.visibility if hasattr(landmark, 'visibility') else 1.0])
                person_data["hand_right_keypoints_2d"] = right_hand_keypoints
            else:
                person_data["hand_right_keypoints_2d"] = [0.0] * 63  # 21 * 3
            
            keypoints_data["people"] = person_data
        
        return keypoints_data
    
    def _mediapipe_to_openpose_pose(self, mediapipe_pose: List[float]) -> List[float]:
        """MediaPipe pose landmarks를 OpenPose 25개 형식으로 변환 (3차원)"""
        # MediaPipe 33개 landmarks -> OpenPose 25개 landmarks 매핑
        openpose_indices = [
            0, 2, 5, 7, 8,  # 0-4: nose, neck, shoulders, elbows
            11, 12, 13, 14, 15, 16,  # 5-10: wrists, hips, knees
            23, 24, 19, 20,  # 11-14: ankles, feet
            1, 15, 16, 17, 18,  # 15-19: face points
            21, 22, 23, 24, 25  # 20-24: additional points
        ]
        
        openpose_pose = []
        for i in range(25):
            if i < len(openpose_indices):
                mp_idx = openpose_indices[i]
                if mp_idx * 3 + 2 < len(mediapipe_pose):
                    openpose_pose.extend([mediapipe_pose[mp_idx * 3], 
                                        mediapipe_pose[mp_idx * 3 + 1],
                                        mediapipe_pose[mp_idx * 3 + 2]])
                else:
                    openpose_pose.extend([0.0, 0.0, 0.0])
            else:
                openpose_pose.extend([0.0, 0.0, 0.0])
        
        return openpose_pose
    
    def _get_face_landmark_indices(self) -> List[int]:
        """MediaPipe face landmarks에서 OpenPose 호환 70개 선택"""
        return list(range(0, 468, 468//70))[:70]
    
    def extract_keypoints_for_dataset(self, data_root: str, split_name: str, 
                                    dataset_name: str) -> bool:
        """
        데이터셋의 비디오들에서 키포인트 추출
        
        Args:
            data_root: 데이터 루트 디렉토리
            split_name: train/val/test
            dataset_name: 데이터셋 이름 (예: "03_syn_word")
        """
        video_base_path = Path(data_root) / split_name / 'video' / f"{dataset_name}_video"
        keypoint_base_path = Path(data_root) / split_name / 'keypoint' / f"{dataset_name}_keypoint"
        
        if not video_base_path.exists():
            logger.error(f"Video directory not found: {video_base_path}")
            return False
        
        # 비디오 디렉토리들 순회
        video_dirs = [d for d in video_base_path.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(video_dirs)} video directories in {video_base_path}")
        
        success_count = 0
        
        for video_dir in video_dirs:
            video_id = video_dir.name
            
            # 이미 키포인트가 있는지 확인
            keypoint_dir = keypoint_base_path / video_id
            if keypoint_dir.exists() and len(list(keypoint_dir.glob('*_keypoints.json'))) > 0:
                logger.info(f"Keypoints already exist for {video_id}, skipping")
                continue
            
            # 비디오 파일 찾기
            video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
            
            if not video_files:
                logger.warning(f"No video files found in {video_dir}")
                continue
            
            video_file = video_files[0]  # 첫 번째 비디오 파일 사용
            
            logger.info(f"Extracting keypoints for {video_id}")
            
            # 키포인트 추출
            if self.extract_from_video(str(video_file), str(keypoint_dir)):
                success_count += 1
                logger.info(f"Successfully extracted keypoints for {video_id}")
            else:
                logger.error(f"Failed to extract keypoints for {video_id}")
        
        logger.info(f"Keypoint extraction completed: {success_count}/{len(video_dirs)} successful")
        return success_count > 0