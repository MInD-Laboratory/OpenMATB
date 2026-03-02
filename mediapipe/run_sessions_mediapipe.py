#!/usr/bin/env python3
"""Run MediaPipe keypoint extraction for OpenMATB session face videos.

This script is a post-task processing step and combines:
- Batch extraction from session facecamera recordings
- Keypoint playback visualization

Extraction uses the MediaPipe Tasks API and requires explicit .task model files.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def _import_external_mediapipe_tasks():
    """Import pip-installed mediapipe tasks even with a local `mediapipe/` folder."""
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    blocked_paths = {script_dir.resolve(), repo_root.resolve()}
    original_sys_path = list(sys.path)

    try:
        filtered: List[str] = []
        for entry in original_sys_path:
            entry_path = Path(entry or ".").resolve()
            if entry_path in blocked_paths:
                continue
            filtered.append(entry)
        sys.path = filtered

        mp_module = importlib.import_module("mediapipe")
        mp_vision = importlib.import_module("mediapipe.tasks.python.vision")
        base_options_module = importlib.import_module("mediapipe.tasks.python.core.base_options")
        return mp_module, mp_vision, base_options_module.BaseOptions
    finally:
        sys.path = original_sys_path


MP, MP_VISION, MP_BASE_OPTIONS = _import_external_mediapipe_tasks()


FACE_FULL_MESH = list(range(478))

FACE_DETAILED = [
    *range(0, 17), 127, 234, 93, 132, 58, 172, 136, 150, 176, 148,
    152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
    70, 63, 105, 66, 107, 55, 65, 52,
    336, 296, 334, 293, 300, 285, 295, 282,
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 99,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]

FACE_KEY_FEATURES = [
    *range(0, 17),
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    168, 197, 195, 94,
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    61, 146, 91, 181, 84, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 317,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
]

FACE_PRESETS = {
    "full-mesh": FACE_FULL_MESH,
    "detailed": FACE_DETAILED,
    "key-features": FACE_KEY_FEATURES,
}

BODY_FULL = list(range(33))
BODY_UPPER = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16,
    23, 24,
]
BODY_POSE = list(range(17))

BODY_PRESETS = {
    "full": BODY_FULL,
    "upper": BODY_UPPER,
    "pose": BODY_POSE,
}

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27),
    (27, 29), (27, 31),
    (24, 26), (26, 28),
    (28, 30), (28, 32),
]


def unique_indices(values: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def output_extension(format_type: str) -> str:
    if format_type == "csv":
        return ".csv"
    if format_type == "parquet":
        return ".parquet"
    if format_type == "hdf5":
        return ".h5"
    if format_type == "pickle":
        return ".pkl"
    raise ValueError(f"Unsupported format: {format_type}")


def save_keypoints(rows: List[List[float]], header: List[str], output_path: Path, format_type: str) -> None:
    frame_data = pd.DataFrame(rows, columns=header)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type == "csv":
        frame_data.to_csv(output_path, index=False)
    elif format_type == "parquet":
        frame_data.to_parquet(output_path, index=False, compression="snappy")
    elif format_type == "hdf5":
        frame_data.to_hdf(output_path, key="keypoints", mode="w", complevel=9)
    elif format_type == "pickle":
        frame_data.to_pickle(output_path, compression="gzip")
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def _safe_confidence(point: object, fallback: float = 1.0) -> float:
    visibility = getattr(point, "visibility", None)
    if visibility is not None:
        return float(visibility)
    presence = getattr(point, "presence", None)
    if presence is not None:
        return float(presence)
    return fallback


def _create_face_landmarker(face_model: Path):
    face_options = MP_VISION.FaceLandmarkerOptions(
        base_options=MP_BASE_OPTIONS(model_asset_path=str(face_model)),
        running_mode=MP_VISION.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return MP_VISION.FaceLandmarker.create_from_options(face_options)


def _create_pose_landmarker(pose_model: Path):
    pose_options = MP_VISION.PoseLandmarkerOptions(
        base_options=MP_BASE_OPTIONS(model_asset_path=str(pose_model)),
        running_mode=MP_VISION.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return MP_VISION.PoseLandmarker.create_from_options(pose_options)


def extract_keypoints_from_video(
    video_path: Path,
    output_path: Path,
    body: bool,
    face: bool,
    body_preset: str,
    face_preset: str,
    format_type: str,
    save_video: bool,
    output_video: Optional[Path],
    show_progress: bool,
    face_model: Optional[Path],
    pose_model: Optional[Path],
) -> None:
    if not body and not face:
        raise ValueError("At least one of body or face must be enabled")

    if face and not face_model:
        raise ValueError("Face extraction requires --face-model <path/to/face_landmarker.task>")
    if body and not pose_model:
        raise ValueError("Body extraction requires --pose-model <path/to/pose_landmarker.task>")

    body_indices = unique_indices(BODY_PRESETS[body_preset]) if body else []
    face_indices = unique_indices(FACE_PRESETS[face_preset]) if face else []

    face_landmarker = _create_face_landmarker(face_model) if face and face_model else None
    pose_landmarker = _create_pose_landmarker(pose_model) if body and pose_model else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if save_video and output_video:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )

    header = ["time"]
    if body:
        for landmark_idx in body_indices:
            header.extend(
                [f"b{landmark_idx}_x", f"b{landmark_idx}_y", f"b{landmark_idx}_z", f"b{landmark_idx}_c"]
            )
    if face:
        for landmark_idx in face_indices:
            header.extend(
                [f"f{landmark_idx}_x", f"f{landmark_idx}_y", f"f{landmark_idx}_z", f"f{landmark_idx}_c"]
            )

    rows: List[List[float]] = []
    progress = tqdm(total=total_frames, desc=f"{video_path.name}", unit="frame") if show_progress else None
    frame_idx = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_sec = frame_idx / fps
            timestamp_ms = int(timestamp_sec * 1000.0)
            row: List[float] = [timestamp_sec]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MP.Image(image_format=MP.ImageFormat.SRGB, data=rgb_frame)

            if body and pose_landmarker:
                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                if pose_result.pose_landmarks:
                    landmarks = pose_result.pose_landmarks[0]
                    for landmark_idx in body_indices:
                        point = landmarks[landmark_idx]
                        row.extend([float(point.x), float(point.y), float(point.z), _safe_confidence(point)])
                else:
                    row.extend([np.nan] * (len(body_indices) * 4))

            if face and face_landmarker:
                face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
                if face_result.face_landmarks:
                    landmarks = face_result.face_landmarks[0]
                    for landmark_idx in face_indices:
                        point = landmarks[landmark_idx]
                        row.extend([float(point.x), float(point.y), float(point.z), _safe_confidence(point)])
                else:
                    row.extend([np.nan] * (len(face_indices) * 4))

            rows.append(row)

            if writer:
                writer.write(frame)

            frame_idx += 1
            if progress:
                progress.update(1)
    finally:
        cap.release()
        if writer:
            writer.release()
        if progress:
            progress.close()
        if face_landmarker:
            face_landmarker.close()
        if pose_landmarker:
            pose_landmarker.close()

    save_keypoints(rows, header, output_path, format_type)


def load_keypoints(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(file_path, key="keypoints")
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(file_path)
    raise ValueError(f"Unsupported keypoints format: {suffix}")


def draw_keypoints_frame(df: pd.DataFrame, frame_idx: int, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if frame_idx >= len(df):
        return canvas

    row = df.iloc[frame_idx]
    body_positions: Dict[int, Tuple[int, int]] = {}

    body_cols = [column for column in df.columns if column.startswith("b") and column.endswith("_x")]
    for x_col in body_cols:
        y_col = x_col.replace("_x", "_y")
        c_col = x_col.replace("_x", "_c")
        if y_col not in df.columns:
            continue
        x_value = row[x_col]
        y_value = row[y_col]
        confidence = row.get(c_col, 1.0)
        if pd.isna(x_value) or pd.isna(y_value) or confidence < 0.5:
            continue
        x_px = max(0, min(width - 1, int(float(x_value) * width)))
        y_px = max(0, min(height - 1, int(float(y_value) * height)))
        landmark_idx = int(x_col.split("_")[0][1:])
        body_positions[landmark_idx] = (x_px, y_px)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx in body_positions and end_idx in body_positions:
            cv2.line(canvas, body_positions[start_idx], body_positions[end_idx], (255, 100, 0), 2, cv2.LINE_AA)

    for x_col in body_cols:
        y_col = x_col.replace("_x", "_y")
        c_col = x_col.replace("_x", "_c")
        if y_col not in df.columns:
            continue
        x_value = row[x_col]
        y_value = row[y_col]
        confidence = row.get(c_col, 1.0)
        if pd.isna(x_value) or pd.isna(y_value) or confidence < 0.5:
            continue
        x_px = max(0, min(width - 1, int(float(x_value) * width)))
        y_px = max(0, min(height - 1, int(float(y_value) * height)))
        cv2.circle(canvas, (x_px, y_px), max(3, int(5 * float(confidence))), (255, 100, 0), -1)

    face_cols = [column for column in df.columns if column.startswith("f") and column.endswith("_x")]
    for x_col in face_cols:
        y_col = x_col.replace("_x", "_y")
        c_col = x_col.replace("_x", "_c")
        if y_col not in df.columns:
            continue
        x_value = row[x_col]
        y_value = row[y_col]
        confidence = row.get(c_col, 1.0)
        if pd.isna(x_value) or pd.isna(y_value) or confidence < 0.5:
            continue
        x_px = max(0, min(width - 1, int(float(x_value) * width)))
        y_px = max(0, min(height - 1, int(float(y_value) * height)))
        cv2.circle(canvas, (x_px, y_px), max(2, int(3 * float(confidence))), (0, 255, 0), -1)

    timestamp = float(row.get("time", frame_idx / 30.0))
    cv2.putText(canvas, f"Frame: {frame_idx}/{len(df)-1}  Time: {timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def playback_keypoints(file_path: Path, speed: float, width: int, height: int, fps: float) -> None:
    frame_data = load_keypoints(file_path)
    delay_ms = int((1000.0 / fps) / speed)

    window_name = f"Keypoint Playback - {file_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    frame_idx = 0
    paused = False
    while True:
        canvas = draw_keypoints_frame(frame_data, frame_idx, width, height)
        if paused:
            cv2.putText(canvas, "PAUSED", (width // 2 - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Speed: {speed}x", (width - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key == ord("r"):
            frame_idx = 0
            paused = False
            continue
        if key == ord("1"):
            speed = 0.5
            delay_ms = int((1000.0 / fps) / speed)
            continue
        if key == ord("2"):
            speed = 1.0
            delay_ms = int((1000.0 / fps) / speed)
            continue
        if key == ord("3"):
            speed = 2.0
            delay_ms = int((1000.0 / fps) / speed)
            continue
        if key == ord("4"):
            speed = 4.0
            delay_ms = int((1000.0 / fps) / speed)
            continue
        if paused and key == 81:
            frame_idx = max(0, frame_idx - 1)
            continue
        if paused and key == 83:
            frame_idx = min(len(frame_data) - 1, frame_idx + 1)
            continue

        if not paused:
            frame_idx += 1
            if frame_idx >= len(frame_data):
                frame_idx = 0

    cv2.destroyAllWindows()


def discover_session_videos(sessions_root: Path) -> List[Tuple[Path, Path]]:
    session_items: List[Tuple[Path, Path]] = []
    for participant_dir in sorted(sessions_root.glob("participant_*")):
        if not participant_dir.is_dir():
            continue
        videos = sorted(participant_dir.glob("*_facecamera.mp4"))
        for video in videos:
            session_items.append((participant_dir, video))
    return session_items


def process_sessions(args: argparse.Namespace) -> None:
    sessions_root = args.sessions_root.resolve()
    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions root not found: {sessions_root}")

    if not args.no_face:
        if not args.face_model:
            raise ValueError("Face extraction is enabled, so --face-model is required")
        if not args.face_model.exists():
            raise FileNotFoundError(f"Face model not found: {args.face_model}")

    if not args.no_body:
        if not args.pose_model:
            raise ValueError("Body extraction is enabled, so --pose-model is required")
        if not args.pose_model.exists():
            raise FileNotFoundError(f"Pose model not found: {args.pose_model}")

    tasks = discover_session_videos(sessions_root)
    if not tasks:
        print(f"[INFO] No facecamera videos found under {sessions_root}")
        return

    done_count = 0
    skipped_count = 0
    failed_count = 0
    participant_status: Dict[str, Dict[str, int]] = {}

    for participant_dir, video_path in tasks:
        participant_key = participant_dir.name
        participant_status.setdefault(participant_key, {"done": 0, "skipped": 0, "failed": 0})

        target_dir = (participant_dir / args.output_dir_name).resolve()
        target_file = target_dir / f"{video_path.stem}_keypoints{output_extension(args.format)}"
        target_video = target_dir / f"{video_path.stem}_visualized.mp4" if args.save_video else None

        if target_file.exists() and target_file.stat().st_size > 0 and not args.force:
            print(f"[SKIP] {participant_key}: {target_file.name} already exists")
            skipped_count += 1
            participant_status[participant_key]["skipped"] += 1
            continue

        print(f"[RUN]  {participant_key}: {video_path.name}")
        try:
            extract_keypoints_from_video(
                video_path=video_path,
                output_path=target_file,
                body=not args.no_body,
                face=not args.no_face,
                body_preset=args.body_preset,
                face_preset=args.face_preset,
                format_type=args.format,
                save_video=args.save_video,
                output_video=target_video,
                show_progress=not args.no_progress,
                face_model=args.face_model,
                pose_model=args.pose_model,
            )
            done_count += 1
            participant_status[participant_key]["done"] += 1
        except Exception as exc:
            failed_count += 1
            participant_status[participant_key]["failed"] += 1
            print(f"[FAIL] {participant_key}: {video_path.name} -> {exc}")

    print("\n[INFO] MediaPipe session processing complete")
    print(f"[INFO] Processed: {done_count}")
    print(f"[INFO] Skipped:   {skipped_count}")
    print(f"[INFO] Failed:    {failed_count}")

    print("\n[INFO] Per participant summary")
    for participant_key in sorted(participant_status):
        status = participant_status[participant_key]
        print(
            f"  {participant_key}: done={status['done']} skipped={status['skipped']} failed={status['failed']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe keypoint processing from OpenMATB sessions (separate post-task step)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process session facecamera videos")
    process_parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path("sessions"),
        help="Root sessions directory (default: sessions)",
    )
    process_parser.add_argument(
        "--output-dir-name",
        type=str,
        default="mediapipe",
        help="Output subdirectory name created under each participant directory (default: mediapipe)",
    )
    process_parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["csv", "parquet", "hdf5", "pickle"],
        help="Output keypoint format (default: parquet)",
    )
    process_parser.add_argument("--no-body", action="store_true", help="Disable body extraction")
    process_parser.add_argument("--no-face", action="store_true", help="Disable face extraction")
    process_parser.add_argument(
        "--body-preset",
        type=str,
        default="full",
        choices=["full", "upper", "pose"],
        help="Body preset (default: full)",
    )
    process_parser.add_argument(
        "--face-preset",
        type=str,
        default="full-mesh",
        choices=["full-mesh", "detailed", "key-features"],
        help="Face preset (default: full-mesh)",
    )
    process_parser.add_argument(
        "--face-model",
        type=Path,
        help="Path to face landmarker .task model file (required when face extraction is enabled)",
    )
    process_parser.add_argument(
        "--pose-model",
        type=Path,
        help="Path to pose landmarker .task model file (required when body extraction is enabled)",
    )
    process_parser.add_argument("--save-video", action="store_true", help="Also save output visualization MP4")
    process_parser.add_argument("--force", action="store_true", help="Reprocess outputs even if already present")
    process_parser.add_argument("--no-progress", action="store_true", help="Disable tqdm frame progress")

    playback_parser = subparsers.add_parser("playback", help="Playback generated keypoint file")
    playback_parser.add_argument("keypoints_file", type=Path, help="Path to keypoints file")
    playback_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        choices=[0.5, 1.0, 2.0, 4.0],
        help="Initial playback speed",
    )
    playback_parser.add_argument("--width", type=int, default=1280, help="Playback width")
    playback_parser.add_argument("--height", type=int, default=720, help="Playback height")
    playback_parser.add_argument("--fps", type=float, default=30.0, help="Reference FPS")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process":
        if args.no_body and args.no_face:
            raise ValueError("Cannot use both --no-body and --no-face")
        process_sessions(args)
        return

    if args.command == "playback":
        if not args.keypoints_file.exists():
            raise FileNotFoundError(f"Keypoints file not found: {args.keypoints_file}")
        playback_keypoints(
            file_path=args.keypoints_file,
            speed=args.speed,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
