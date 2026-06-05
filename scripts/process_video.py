import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def process_video(source_path: str, model_path: str, output_path: str, conf: float = 0.35):
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=conf, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # class 0 thường là person/human
                if cls_id != 0:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Person {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        writer.write(frame)
        frame_index += 1

        if total_frames > 0:
            progress = frame_index / total_frames * 100
            print(f"\rProcessing: {progress:.1f}%", end="")

    cap.release()
    writer.release()

    print(f"\nDone. Output saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Process video with YOLOv8 human detection")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--output", default="outputs/videos/result.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")

    args = parser.parse_args()

    process_video(
        source_path=args.source,
        model_path=args.model,
        output_path=args.output,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()