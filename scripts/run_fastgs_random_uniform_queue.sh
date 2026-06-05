#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
EXPERIMENT_ID="${EXPERIMENT_ID:-0001_stage1_register_quality_gate}"
PREPARED_ROOT="${PREPARED_ROOT:-$PROJECT_ROOT/runs/$EXPERIMENT_ID/prepared/3dgsdata}"
QUEUE_ROOT="${QUEUE_ROOT:-$PROJECT_ROOT/runs/$EXPERIMENT_ID/queues/random_uniform_images4_30k}"
FASTGS_ROOT="${FASTGS_ROOT:-$PROJECT_ROOT/external/FastGS}"
FASTGS_PYTHON="${FASTGS_PYTHON:-$FASTGS_ROOT/.venv/bin/python}"
ITERATIONS="${ITERATIONS:-30000}"
DENSIFICATION_INTERVAL="${DENSIFICATION_INTERVAL:-100}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-fastgs_output_images4_30k}"
SESSION="${SESSION:-fastgs_ru_3dgsdata}"

JOBS_TSV="$QUEUE_ROOT/jobs.tsv"
LOG_DIR="$QUEUE_ROOT/logs"
CLAIMS_DIR="$QUEUE_ROOT/claims"
DONE_DIR="$QUEUE_ROOT/done"
FAILED_DIR="$QUEUE_ROOT/failed"

usage() {
  cat <<EOF
Usage: $0 <prepare|launch|worker|status>

Environment overrides:
  PROJECT_ROOT=$PROJECT_ROOT
  PREPARED_ROOT=$PREPARED_ROOT
  QUEUE_ROOT=$QUEUE_ROOT
  FASTGS_PYTHON=$FASTGS_PYTHON
  ITERATIONS=$ITERATIONS
  DENSIFICATION_INTERVAL=$DENSIFICATION_INTERVAL
  MODEL_DIR_NAME=$MODEL_DIR_NAME
  SESSION=$SESSION
EOF
}

init_dirs() {
  mkdir -p "$QUEUE_ROOT" "$LOG_DIR" "$CLAIMS_DIR" "$DONE_DIR" "$FAILED_DIR"
}

prepare_jobs() {
  init_dirs
  "$FASTGS_PYTHON" - "$PREPARED_ROOT" "$MODEL_DIR_NAME" "$JOBS_TSV" <<'PY'
import json
import sys
from pathlib import Path

prepared_root = Path(sys.argv[1])
model_dir_name = sys.argv[2]
jobs_tsv = Path(sys.argv[3])
image_suffixes = {".jpg", ".jpeg", ".png"}


def image_stem(name: str) -> str:
    return Path(name).stem


def first_source_target(source_dir: Path) -> Path:
    image_dir = source_dir / "images"
    for image_path in sorted(image_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in image_suffixes:
            return image_path.resolve()
    raise RuntimeError(f"{source_dir}: no source images found")


def scene_image_dir(source_dir: Path) -> Path:
    target = first_source_target(source_dir)
    parts = target.parts
    if "images" not in parts:
        raise RuntimeError(f"{source_dir}: cannot locate original images dir from {target}")
    index = len(parts) - 1 - list(reversed(parts)).index("images")
    return Path(*parts[: index + 1])


def full_llffhold_test_stems(source_dir: Path, llffhold: int = 8) -> set[str]:
    images_dir = scene_image_dir(source_dir)
    full_names = sorted(
        (path.name for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in image_suffixes),
        key=lambda name: Path(name).stem,
    )
    return {image_stem(name) for order, name in enumerate(full_names) if order % llffhold == 0}


def validate_split(manifest_path: Path, manifest: dict, source_dir: Path) -> None:
    split_path = source_dir / "stage1_split.json"
    if not split_path.exists():
        raise RuntimeError(f"{manifest_path}: missing {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_images = split.get("train_images") or []
    test_images = split.get("test_images") or []
    train_stems = {image_stem(name) for name in train_images}
    test_stems = {image_stem(name) for name in test_images}

    if not train_images or not test_images:
        raise RuntimeError(f"{manifest_path}: split must contain non-empty train and test images")
    if train_stems.intersection(test_stems):
        raise RuntimeError(f"{manifest_path}: train/test split overlaps")
    if len(train_images) != int(manifest.get("selected_count", -1)):
        raise RuntimeError(
            f"{manifest_path}: train_images={len(train_images)} selected_count={manifest.get('selected_count')}"
        )
    if len(test_images) != int(manifest.get("heldout_test_count", -1)):
        raise RuntimeError(
            f"{manifest_path}: test_images={len(test_images)} heldout_test_count={manifest.get('heldout_test_count')}"
        )

    expected_test_stems = full_llffhold_test_stems(source_dir)
    if test_stems != expected_test_stems:
        missing = sorted(expected_test_stems.difference(test_stems))[:8]
        extra = sorted(test_stems.difference(expected_test_stems))[:8]
        raise RuntimeError(
            f"{manifest_path}: test set is not the full-scene llffhold split; "
            f"missing={missing} extra={extra}"
        )
    if not train_stems.isdisjoint(expected_test_stems):
        overlap = sorted(train_stems.intersection(expected_test_stems))[:8]
        raise RuntimeError(f"{manifest_path}: selected train contains heldout full-scene test images: {overlap}")

jobs = []
errors = []
for manifest_path in sorted(prepared_root.glob("*/**/ratio_020/stage1_subset_manifest.json")):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_method = manifest.get("base_method")
    method = manifest.get("method", "")
    if base_method not in {"random_ratio", "uniform_stride_ratio"} and method != "uniform_stride_ratio":
        continue
    if manifest.get("status") != "ready":
        continue

    run_dir = manifest_path.parent
    source_dir = run_dir / "fastgs_source"
    if not source_dir.exists():
        continue
    try:
        validate_split(manifest_path, manifest, source_dir)
    except Exception as exc:
        errors.append(str(exc))
        continue
    scene_id = manifest["scene_id"]
    job_id = f"{scene_id}__{method}"
    model_path = run_dir / model_dir_name
    jobs.append((scene_id, method, job_id, run_dir, source_dir, model_path))

if errors:
    print("Split validation failed; refusing to write queue.", file=sys.stderr)
    for error in errors[:40]:
        print(f"- {error}", file=sys.stderr)
    if len(errors) > 40:
        print(f"... {len(errors) - 40} more errors", file=sys.stderr)
    raise SystemExit(1)

jobs_tsv.parent.mkdir(parents=True, exist_ok=True)
with jobs_tsv.open("w", encoding="utf-8") as f:
    f.write("scene_id\tmethod\tjob_id\trun_dir\tsource_dir\tmodel_path\n")
    for row in jobs:
        f.write("\t".join(str(value) for value in row) + "\n")

print(f"Wrote {len(jobs)} jobs to {jobs_tsv}")
PY
}

ensure_images4() {
  local source_dir="$1"
  local source_images="$source_dir/images"
  local images4="$source_dir/images_4"
  local first_image
  local first_target
  local scene_root
  local native_images4

  first_image="$(find "$source_images" -type f -o -type l | sort | head -n 1)"
  if [[ -z "$first_image" ]]; then
    echo "No source images found under $source_images" >&2
    return 1
  fi

  first_target="$(readlink -f "$first_image")"
  scene_root="${first_target%/images/*}"
  native_images4="$scene_root/images_4"

  if [[ -d "$native_images4" ]]; then
    if [[ -L "$images4" || ! -e "$images4" ]]; then
      ln -sfn "$native_images4" "$images4"
    fi
    echo "images_4:native:$native_images4"
    return 0
  fi

  if [[ -L "$images4" ]]; then
    rm "$images4"
  fi
  mkdir -p "$images4"
  "$FASTGS_PYTHON" - "$source_images" "$images4" <<'PY'
import sys
from pathlib import Path
from PIL import Image

src_root = Path(sys.argv[1])
dst_root = Path(sys.argv[2])
suffixes = {".jpg", ".jpeg", ".png"}
count = 0

for src in sorted(src_root.rglob("*")):
    if not src.is_file() or src.suffix.lower() not in suffixes:
        continue
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        continue
    with Image.open(src) as image:
        image = image.convert("RGB")
        width, height = image.size
        resized = image.resize((max(1, width // 4), max(1, height // 4)), Image.Resampling.LANCZOS)
        save_kwargs = {}
        if dst.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update(quality=95, subsampling=0)
        resized.save(dst, **save_kwargs)
    count += 1

(dst_root / ".generated_factor4").write_text(f"generated_or_refreshed={count}\n", encoding="utf-8")
print(f"images_4:generated:{dst_root}:{count}")
PY
}

run_job() {
  local gpu="$1"
  local scene_id="$2"
  local method="$3"
  local job_id="$4"
  local run_dir="$5"
  local source_dir="$6"
  local model_path="$7"
  local job_log="$LOG_DIR/$job_id.worker.log"
  local train_log="$model_path/train.log"
  local render_log="$model_path/render_test.log"
  local metrics_log="$model_path/metrics.log"
  local started_at
  local finished_at

  mkdir -p "$model_path"
  started_at="$(date --iso-8601=seconds)"
  {
    echo "[$started_at] START gpu=$gpu scene=$scene_id method=$method"
    echo "run_dir=$run_dir"
    echo "source_dir=$source_dir"
    echo "model_path=$model_path"
    ensure_images4 "$source_dir"
  } >> "$job_log" 2>&1

  if [[ ! -f "$model_path/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$FASTGS_PYTHON" "$FASTGS_ROOT/train.py" \
      --source_path "$source_dir" \
      --model_path "$model_path" \
      --images images_4 \
      --eval \
      --iterations "$ITERATIONS" \
      --densification_interval "$DENSIFICATION_INTERVAL" \
      --test_iterations "$ITERATIONS" \
      --save_iterations "$ITERATIONS" \
      --checkpoint_iterations -1 \
      > "$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] SKIP train, point cloud already exists" >> "$job_log"
  fi

  if [[ ! -d "$model_path/test/ours_${ITERATIONS}/renders" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$FASTGS_PYTHON" "$FASTGS_ROOT/render.py" \
      --model_path "$model_path" \
      --iteration "$ITERATIONS" \
      --skip_train \
      > "$render_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] SKIP render, renders already exist" >> "$job_log"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "$FASTGS_PYTHON" "$FASTGS_ROOT/metrics.py" \
    -m "$model_path" \
    > "$metrics_log" 2>&1

  finished_at="$(date --iso-8601=seconds)"
  {
    echo "[$finished_at] DONE gpu=$gpu scene=$scene_id method=$method"
    grep -E "SSIM|PSNR|LPIPS" "$metrics_log" || true
  } >> "$job_log"
}

worker() {
  local gpu="${1:?GPU id is required}"
  local worker_log="$LOG_DIR/worker_gpu${gpu}.log"
  init_dirs
  if [[ ! -f "$JOBS_TSV" ]]; then
    prepare_jobs
  fi

  echo "[$(date --iso-8601=seconds)] worker gpu=$gpu started" >> "$worker_log"
  tail -n +2 "$JOBS_TSV" | while IFS=$'\t' read -r scene_id method job_id run_dir source_dir model_path; do
    [[ -n "$job_id" ]] || continue
    if [[ -f "$DONE_DIR/$job_id" ]]; then
      continue
    fi
    if [[ -f "$model_path/results.json" ]]; then
      echo "[$(date --iso-8601=seconds)] mark existing done $job_id" >> "$worker_log"
      touch "$DONE_DIR/$job_id"
      continue
    fi
    if mkdir "$CLAIMS_DIR/$job_id" 2>/dev/null; then
      echo "[$(date --iso-8601=seconds)] claimed $job_id" >> "$worker_log"
      if run_job "$gpu" "$scene_id" "$method" "$job_id" "$run_dir" "$source_dir" "$model_path"; then
        touch "$DONE_DIR/$job_id"
        rm -f "$FAILED_DIR/$job_id"
        echo "[$(date --iso-8601=seconds)] finished $job_id" >> "$worker_log"
      else
        echo "[$(date --iso-8601=seconds)] failed $job_id" >> "$worker_log"
        date --iso-8601=seconds > "$FAILED_DIR/$job_id"
      fi
    fi
  done
  echo "[$(date --iso-8601=seconds)] worker gpu=$gpu exhausted queue" >> "$worker_log"
}

launch() {
  prepare_jobs
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION already exists"
    status
    return 0
  fi
  tmux new-session -d -s "$SESSION" -n gpu0 "cd '$PROJECT_ROOT' && bash '$0' worker 0"
  tmux new-window -t "${SESSION}:" -n gpu1 "cd '$PROJECT_ROOT' && bash '$0' worker 1"
  echo "launched tmux session $SESSION with GPU workers 0 and 1"
}

status() {
  init_dirs
  local total done failed claimed running
  total=0
  if [[ -f "$JOBS_TSV" ]]; then
    total=$(( $(wc -l < "$JOBS_TSV") - 1 ))
  fi
  done="$(find "$DONE_DIR" -maxdepth 1 -type f | wc -l)"
  failed="$(find "$FAILED_DIR" -maxdepth 1 -type f | wc -l)"
  claimed="$(find "$CLAIMS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  running=$(( claimed - done - failed ))
  if (( running < 0 )); then
    running=0
  fi
  echo "queue=$QUEUE_ROOT"
  echo "jobs_total=$total done=$done failed=$failed claimed=$claimed running_or_claimed=$running"
  tmux list-windows -t "$SESSION" 2>/dev/null || true
  for log in "$LOG_DIR"/worker_gpu*.log; do
    [[ -f "$log" ]] || continue
    echo "--- ${log#$PROJECT_ROOT/}"
    tail -n 8 "$log"
  done
}

cmd="${1:-}"
case "$cmd" in
  prepare)
    prepare_jobs
    ;;
  launch)
    launch
    ;;
  worker)
    shift
    worker "$@"
    ;;
  status)
    status
    ;;
  *)
    usage
    exit 2
    ;;
esac
