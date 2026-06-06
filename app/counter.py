from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import cv2


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

    def _get_side(self, center: Tuple[int, int]) -> str:
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