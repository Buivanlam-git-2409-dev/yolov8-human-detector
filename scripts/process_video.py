"""Video processing pipeline for Smart Store People Counting System."""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
from ultralytics import YOLO


# PeopleCounter class - defined here to avoid import issues
@dataclass
class CrossingEvent:
    track_id: int
    direction: str
    frame_index: int
    timestamp_seconds: float


@dataclass
class PeopleCounter:
    line_orientation: str = "horizontal"
    line_position: int = 360
    cooldown_frames: int = 30

    previous_side: Dict[int, str] = field(default_factory=dict)
    last_counted_frame: Dict[int, int] = field(default_factory=dict)
    in_count: int = 0
    out_count: int = 0
    event_log: List[CrossingEvent] = field(default_factory=list)

    def _get_side(self, center: tuple) -> str:
        cx, cy = center

        if self.line_orientation == "horizontal":
            return "above" if cy < self.line_position else "below"

        return "left" if cx < self.line_position else "right"

    def _get_direction(self, old_side: str, new_side: str) -> Optional[str]:
        if self.line_orientation == "horizontal":
            if old_side == "above" and new_side == "below":
                return "IN"
            if old_side == "below" and new_side == "above":
                return "OUT"

        if self.line_orientation == "vertical":
            if old_side == "left" and new_side == "right":
                return "IN"
            if old_side == "right" and new_side == "left":
                return "OUT"

        return None

    def update(
        self,
        tracked_objects: List[dict],
        frame_index: int,
        fps: float,
    ) -> None:
        timestamp_seconds = frame_index / fps if fps > 0 else 0

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            center = obj.get("center")

            if track_id is None or center is None:
                continue

            track_id = int(track_id)
            center = tuple(map(int, center))

            new_side = self._get_side(center)

            if track_id in self.previous_side:
                old_side = self.previous_side[track_id]
                direction = self._get_direction(old_side, new_side)

                if direction:
                    last_frame = self.last_counted_frame.get(track_id, -999999)

                    if frame_index - last_frame >= self.cooldown_frames:
                        if direction == "IN":
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        self.last_counted_frame[track_id] = frame_index
                        self.event_log.append(
                            CrossingEvent(
                                track_id=track_id,
                                direction=direction,
                                frame_index=frame_index,
                                timestamp_seconds=round(timestamp_seconds, 2),
                            )
                        )

            self.previous_side[track_id] = new_side

    @property
    def current_occupancy(self) -> int:
        return max(0, self.in_count - self.out_count)

    def draw(self, frame):
        height, width = frame.shape[:2]

        if self.line_orientation == "horizontal":
            y = int(self.line_position)
            cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2)
            cv2.putText(frame, "IN", (20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "OUT", (20, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            x = int(self.line_position)
            cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2)
            cv2.putText(frame, "IN", (x + 15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "OUT", (x - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (10, 10), (360, 115), (0, 0, 0), -1)
        cv2.putText(frame, f"IN: {self.in_count}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {self.out_count}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"NOW: {self.current_occupancy}", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    def get_event_log_as_dicts(self) -> List[dict]:
        return [
            {
                "track_id": event.track_id,
                "direction": event.direction,
                "frame_index": event.frame_index,
                "timestamp_seconds": event.timestamp_seconds,
            }
            for event in self.event_log
        ]


def get_video_metadata(video_path: str) -> dict:
    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
    }


def save_event_log_csv(event_log: List[dict], csv_path: str) -> str:
    """Save crossing events to CSV and return the saved path."""
    output = Path(csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["track_id", "direction", "frame_index", "timestamp_seconds"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_log)

    return str(output)


def process_video(
    source_path: str,
    model_path: str,
    output_path: str,
    conf: float = 0.35,
    iou: float = 0.45,
    tracker: str = "bytetrack.yaml",
    line_orientation: str = "horizontal",
    line_position: int = 360,
    cooldown_frames: int = 30,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, object]:
    """Process a video, track people, count line crossings, and save annotated output."""
    source = Path(source_path)
    model_file = Path(model_path)
    output = Path(output_path)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_file))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video size.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    counter = PeopleCounter(
        line_orientation=line_orientation,
        line_position=line_position,
        cooldown_frames=cooldown_frames,
    )

    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                conf=conf,
                iou=iou,
                verbose=False,
            )

            tracked_objects = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    # Class 0 is usually person/human for YOLO person models.
                    if cls_id != 0:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                    track_id = int(boxes.id[i]) if boxes.id is not None else None
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    tracked_objects.append(
                        {
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "center": (cx, cy),
                            "confidence": confidence,
                            "class_id": cls_id,
                        }
                    )

                    label = f"ID {track_id} Person {confidence:.2f}" if track_id is not None else f"Person {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            counter.update(tracked_objects, frame_index, fps)
            frame = counter.draw(frame)

            writer.write(frame)
            frame_index += 1

            if total_frames > 0:
                progress = min(frame_index / total_frames, 1.0)
                if progress_callback:
                    progress_callback(progress)
                else:
                    print(f"\rProcessing: {progress * 100:.1f}%", end="")
    finally:
        cap.release()
        writer.release()

    if not progress_callback:
        print(f"\nDone. Output saved to: {output}")
        print(f"IN: {counter.in_count} | OUT: {counter.out_count} | NOW: {counter.current_occupancy}")

    return {
        "output_path": str(output),
        "in_count": counter.in_count,
        "out_count": counter.out_count,
        "current_occupancy": counter.current_occupancy,
        "event_log": counter.get_event_log_as_dicts(),
        "processed_frames": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process video with YOLOv8 tracking and line-crossing people counting"
    )
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--output", default="outputs/videos/tracking_result.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--line-orientation", default="horizontal", choices=["horizontal", "vertical"])
    parser.add_argument("--line-position", type=int, default=360, help="Line position: y for horizontal, x for vertical")
    parser.add_argument("--crowd-threshold", type=int, default=10, help="Occupancy threshold for crowd warning")
    parser.add_argument("--save-csv", action="store_true", help="Save event log CSV next to output video")

    args = parser.parse_args()

    result = process_video(
        source_path=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        line_orientation=args.line_orientation,
        line_position=args.line_position,
    )

    if args.save_csv:
        csv_path = str(Path(args.output).with_suffix(".csv"))
        save_event_log_csv(result["event_log"], csv_path)
        print(f"CSV saved to: {csv_path}")

    status = "WARNING" if result["current_occupancy"] >= args.crowd_threshold else "NORMAL"
    print(f"Crowd status: {status}")


if __name__ == "__main__":
    main()
