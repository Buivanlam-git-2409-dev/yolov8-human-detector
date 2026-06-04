# CLAUDE.md
<!--
Giữ ngắn — dưới 150 dòng.
Chỉ include những gì Claude không thể tự biết.
Test từng rule: nếu Claude đã làm đúng mà không cần → xóa.
-->

## Project
Smart Store People Counting System using YOLOv8.
Nâng cấp từ app Human Detector chung chung thành hệ thống đếm người vào/ra cửa hàng bằng YOLOv8 + tracking + line crossing.
Mục tiêu chính: upload video, detect người, gán tracking ID, đếm IN/OUT khi người đi qua line, hiển thị dashboard và xuất video/CSV.

## Stack
- Backend / CV: Python, Ultralytics YOLOv8, OpenCV, NumPy, Pandas
- Frontend: Streamlit
- Model: `models/best.pt`
- Tracking: ByteTrack hoặc BOT-SORT qua Ultralytics `model.track()`
- Infra: GitHub, local development; chưa cần Docker nếu không được yêu cầu

## Commands
```bash
# Development
pip install -r requirements.txt        # Install dependencies
streamlit run app/app.py               # Start Streamlit app
python scripts/process_video.py --help # Show CLI options

# Example CLI video processing
python scripts/process_video.py \
  --source sample_videos/store_entrance.mp4 \
  --model models/best.pt \
  --output outputs/videos/result.mp4 \
  --line-orientation horizontal \
  --line-position 360 \
  --tracker bytetrack.yaml \
  --conf 0.35

# Testing / quality
python -m pytest                        # Run tests if tests exist
python -m compileall app scripts        # Quick syntax check
```

## Conventions
- Branch naming: `feature/`, `fix/`, `chore/` prefix.
- Commit format: `type(scope): description`, e.g. `feat(counter): add line crossing logic`.
- Do not commit generated videos, large model files, cache folders, or local env files.
- Use relative paths only. Never hardcode absolute Windows paths like `C:\Users\...`.
- Keep functions small, readable, and documented with short docstrings.
- Prefer simple, reliable MVP over complex logic.

## Architecture notes
- Keep Streamlit UI in `app/app.py` only. Do not put business logic directly in UI code.
- Put video processing in `app/video_processor.py`.
- Put line crossing logic in `app/counter.py`.
- Put shared helpers in `app/utils.py`.
- Optional CLI entrypoint should be in `scripts/process_video.py` and reuse app modules.
- Pipeline: Video Input → YOLOv8 Tracking → Tracked Objects → Line Crossing Counter → Annotated Video + Metrics + CSV.

## Counting logic
- Use `model.track(frame, persist=True, tracker="bytetrack.yaml")` for video tracking.
- Only count human/person class. Assume class `0` is person/human unless `model.names` proves otherwise.
- For each tracked object, store `track_id`, bbox, confidence, class_id, and center point.
- Horizontal line: compare previous `cy` and current `cy` with `line_y`.
- Vertical line: compare previous `cx` and current `cx` with `line_x`.
- Count IN/OUT only when a track center crosses the configured line.
- Avoid duplicate counting with crossing state or cooldown per `track_id`.
- `current_occupancy = max(0, in_count - out_count)`.

## Streamlit requirements
- App title: `Smart Store People Counting System`.
- Keep image detection working if already present.
- Add video upload for `.mp4`, `.avi`, `.mov`.
- Sidebar settings:
  - model path
  - confidence threshold
  - IOU threshold
  - tracker type: `bytetrack.yaml` / `botsort.yaml`
  - line orientation: horizontal / vertical
  - line position
  - crowd threshold
  - show track trails checkbox
- Main dashboard must show:
  - People In
  - People Out
  - Current Occupancy
  - Crowd Alert: Normal / Warning
- Provide download buttons for annotated video and CSV event log.

## Output rules
- Save annotated videos to `outputs/videos/`.
- Save CSV logs to `outputs/logs/`.
- CSV columns: `track_id,direction,frame_index,timestamp_seconds`.
- Create output folders automatically if missing.
- Do not fail silently. Show useful error messages when model/video cannot be loaded.

## YOU MUST
- Run a syntax check before saying a task is complete.
- Do not commit directly to `main`; create/use a feature branch.
- Do not remove existing working image detection unless explicitly asked.
- Do not retrain the model unless explicitly asked.
- Do not introduce heavy dependencies without asking.
- Do not hardcode local absolute paths.

---

## Current State
<!-- Claude cập nhật mỗi session — không xóa section này -->
- Last session: [ngày] — [những gì đã xong]
- In progress: [feature/task đang làm dở]
- Next: [bước tiếp theo cụ thể]

## Key Decisions
<!-- Không xóa — append khi có decision mới -->
- [2026-06-04] — Chọn hướng nâng cấp thành Smart Store People Counting thay vì Human Detector chung chung.
- [2026-06-04] — MVP ưu tiên video upload + YOLO tracking + line crossing + Streamlit dashboard + CSV export.
