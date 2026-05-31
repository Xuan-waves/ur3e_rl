#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${UR3E_RLT_CONDA_ENV:-ur3e_rlt}"

config_value() {
    local attr="$1"
    (cd "$REPO_ROOT" && python3 -c "from scripts.collect_data.config import CollectConfig; print(getattr(CollectConfig, '$attr'))")
}

DATASET_ROOT="$(config_value dataset_root)"
DATASET_NAME="$(config_value dataset_name)"
TASK="$(config_value task)"
PREVIEW_TOPIC="$(config_value preview_topic)"
REFERENCE_CAMERA="$(config_value reference_camera)"
LAUNCH_REALSENSE_ROS=0
CAMERA_SOURCE="$(config_value camera_source)"
WITH_PREVIEW=1
STATUS_PANEL=1
STATUS_HZ="$(config_value status_hz)"
MAX_EPISODES="$(config_value max_episodes)"
D455_SERIAL=""
D405_SERIAL=""
D455_PORT=""
D405_PORT=""
CAMERA_WIDTH="$(config_value camera_width)"
CAMERA_HEIGHT="$(config_value camera_height)"
CAMERA_FPS="$(config_value camera_fps)"
MAX_DT_FRONT_IMAGE="$(config_value max_dt_front_image)"
MAX_DT_WRIST_IMAGE="$(config_value max_dt_wrist_image)"
MAX_DT_STATE="$(config_value max_dt_state)"
MAX_DT_ACTION="$(config_value max_dt_action)"
ALLOW_STALE_FRONT=0

usage() {
    cat <<'EOF'
Usage: scripts/collect_data/run_collect_data_tabs.sh [options]

Starts GNOME Terminal tabs:
  1. UR3e LeRobot collector subscribed to ROS camera topics
  2. OpenCV preview subscribed to the raw ROS camera topics

By default this follows the reference project: camera publishers are treated as
an external precondition and are not started by this launcher.  Start and verify
RealSense ROS separately before collecting.

Only the RealSense ROS nodes open the devices.  The collector and preview are
normal ROS subscribers, keeping visualization separate from collection.

Options:
  --conda-env NAME       Conda env. Default: $UR3E_RLT_CONDA_ENV or ur3e_rlt.
  --dataset-root PATH    Dataset parent directory. Default: CollectConfig.dataset_root.
  --dataset-name NAME    Dataset session prefix. Default: CollectConfig.dataset_name.
  --task TEXT            Task text saved to LeRobot. Default: CollectConfig.task.
  --max-episodes N       Stop after saving N episodes. 0 means unlimited.
  --d455-serial SERIAL   D455 serial. Auto-detected if omitted.
  --d405-serial SERIAL   D405 serial. Auto-detected if omitted.
  --d455-port PORT       D455 USB port id, e.g. 2-1. Auto-detected if omitted.
  --d405-port PORT       D405 USB port id, e.g. 2-2. Auto-detected if omitted.
  --camera-width WIDTH   Default: 640.
  --camera-height HEIGHT Default: 480.
  --camera-fps FPS       Default: 30.
  --preview-topic TOPIC  Default: /ur3e_vr/collection_preview.
  --reference-camera front|wrist
                        Camera used as data synchronization reference. Default: wrist.
  --camera-source realsense|ros
                        Default: ros. realsense is kept only as an experimental direct path.
  --max-dt-front SEC    D455/front nearest-image tolerance. Default: 0.08.
  --max-dt-wrist SEC    D405/wrist nearest-image tolerance. Default: 0.08.
  --max-dt-state SEC    Robot state tolerance. Default: 0.08.
  --max-dt-action SEC   IK target and VR command tolerance. Default: 0.08.
  --allow-stale-front   Reuse the latest D455/front frame when debugging only.
  --launch-realsense-ros
                        Also open a camera tab using this project's RealSense helper.
  --no-launch-realsense-ros
                        Use already-running RealSense ROS camera publishers. Default.
  --no-preview          Do not open the OpenCV preview tab.
  --no-status-panel     Disable the in-place collector status panel.
  --status-hz HZ        Status panel refresh rate. Default: 4.0.
EOF
}

quote() {
    printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --max-episodes)
            MAX_EPISODES="$2"
            shift 2
            ;;
        --d455-serial)
            D455_SERIAL="$2"
            shift 2
            ;;
        --d405-serial)
            D405_SERIAL="$2"
            shift 2
            ;;
        --d455-port)
            D455_PORT="$2"
            shift 2
            ;;
        --d405-port)
            D405_PORT="$2"
            shift 2
            ;;
        --camera-width)
            CAMERA_WIDTH="$2"
            shift 2
            ;;
        --camera-height)
            CAMERA_HEIGHT="$2"
            shift 2
            ;;
        --camera-fps)
            CAMERA_FPS="$2"
            shift 2
            ;;
        --preview-topic)
            PREVIEW_TOPIC="$2"
            shift 2
            ;;
        --reference-camera)
            REFERENCE_CAMERA="$2"
            shift 2
            ;;
        --camera-source)
            CAMERA_SOURCE="$2"
            shift 2
            ;;
        --max-dt-front)
            MAX_DT_FRONT_IMAGE="$2"
            shift 2
            ;;
        --max-dt-wrist)
            MAX_DT_WRIST_IMAGE="$2"
            shift 2
            ;;
        --max-dt-state)
            MAX_DT_STATE="$2"
            shift 2
            ;;
        --max-dt-action)
            MAX_DT_ACTION="$2"
            shift 2
            ;;
        --strict-front)
            ALLOW_STALE_FRONT=0
            shift
            ;;
        --allow-stale-front)
            ALLOW_STALE_FRONT=1
            shift
            ;;
        --launch-realsense-ros)
            LAUNCH_REALSENSE_ROS=1
            shift
            ;;
        --no-launch-realsense-ros)
            LAUNCH_REALSENSE_ROS=0
            shift
            ;;
        --no-preview)
            WITH_PREVIEW=0
            shift
            ;;
        --no-status-panel)
            STATUS_PANEL=0
            shift
            ;;
        --status-hz)
            STATUS_HZ="$2"
            shift 2
            ;;
        --no-rqt)
            WITH_PREVIEW=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "[collect-launcher] gnome-terminal not found." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PID_DIR="/tmp/ur3e_collect_tabs_${RUN_ID}"
mkdir -p "$PID_DIR"

create_runner() {
    local runner_path="$1"
    local title="$2"
    local body="$3"

    cat > "$runner_path" <<EOF
#!/usr/bin/env bash
set +e
set -m

REPO_ROOT=$(quote "$REPO_ROOT")
CONDA_ENV_NAME=$(quote "$CONDA_ENV_NAME")
TITLE=$(quote "$title")

activate_conda_env() {
    local env_name="\$1"
    if command -v conda >/dev/null 2>&1; then
        eval "\$(conda shell.bash hook)" || true
    elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[collect-launcher] Could not find conda." >&2
        return 1
    fi
    conda activate "\${env_name}"
}

cd "\${REPO_ROOT}" || exit 1
echo "╭─ \${TITLE}"
echo "│ working dir: \${REPO_ROOT}"
echo "│ conda env  : \${CONDA_ENV_NAME}"
echo "╰────────────────────────────────────────────────────────"

if ! activate_conda_env "\${CONDA_ENV_NAME}"; then
    echo "[collect-launcher] Failed to activate conda environment: \${CONDA_ENV_NAME}" >&2
    exec bash
fi

$body
exit_code=\$?
echo
echo "[collect-launcher] \${TITLE} exited with code \${exit_code}"
echo "[collect-launcher] Press Ctrl+D or type exit to close this tab."
exec bash
EOF

    chmod +x "$runner_path"
}

COLLECTOR_CMD="set +u; source /opt/ros/humble/setup.bash; set -u; python scripts/collect_data/collect_ur3e_vr_impedance.py --camera-source $(quote "$CAMERA_SOURCE") --dataset-root $(quote "$DATASET_ROOT") --dataset-name $(quote "$DATASET_NAME") --task $(quote "$TASK") --preview-topic $(quote "$PREVIEW_TOPIC") --no-preview-window --reference-camera $(quote "$REFERENCE_CAMERA") --camera-width $(quote "$CAMERA_WIDTH") --camera-height $(quote "$CAMERA_HEIGHT") --camera-fps $(quote "$CAMERA_FPS") --max-dt-front-image $(quote "$MAX_DT_FRONT_IMAGE") --max-dt-wrist-image $(quote "$MAX_DT_WRIST_IMAGE") --max-dt-state $(quote "$MAX_DT_STATE") --max-dt-action $(quote "$MAX_DT_ACTION") --max-episodes $(quote "$MAX_EPISODES") --status-hz $(quote "$STATUS_HZ") --no-launch-realsense-ros"
if [[ "$STATUS_PANEL" -eq 1 ]]; then
    COLLECTOR_CMD+=" --status-panel"
else
    COLLECTOR_CMD+=" --no-status-panel"
fi
if [[ "$ALLOW_STALE_FRONT" -eq 1 ]]; then
    COLLECTOR_CMD+=" --allow-stale-front"
else
    COLLECTOR_CMD+=" --strict-front"
fi
if [[ -n "$D455_SERIAL" ]]; then
    COLLECTOR_CMD+=" --front-camera-serial $(quote "$D455_SERIAL")"
fi
if [[ -n "$D405_SERIAL" ]]; then
    COLLECTOR_CMD+=" --wrist-camera-serial $(quote "$D405_SERIAL")"
fi

COLLECTOR_RUNNER="$PID_DIR/collector.sh"
PREVIEW_RUNNER="$PID_DIR/preview.sh"
CAMERA_RUNNER="$PID_DIR/cameras.sh"

CAMERA_CMD="scripts/collect_data/run_realsense_cameras.sh --width $(quote "$CAMERA_WIDTH") --height $(quote "$CAMERA_HEIGHT") --fps $(quote "$CAMERA_FPS")"
if [[ -n "$D455_SERIAL" ]]; then
    CAMERA_CMD+=" --d455-serial $(quote "$D455_SERIAL")"
fi
if [[ -n "$D405_SERIAL" ]]; then
    CAMERA_CMD+=" --d405-serial $(quote "$D405_SERIAL")"
fi
if [[ -n "$D455_PORT" ]]; then
    CAMERA_CMD+=" --d455-port $(quote "$D455_PORT")"
fi
if [[ -n "$D405_PORT" ]]; then
    CAMERA_CMD+=" --d405-port $(quote "$D405_PORT")"
fi
create_runner "$COLLECTOR_RUNNER" "UR3e LeRobot Collector" "$COLLECTOR_CMD"
PREVIEW_CMD="set +u; source /opt/ros/humble/setup.bash; set -u; python scripts/collect_data/preview_collection_topic.py --front-topic /camera/d455/color/image_raw --wrist-topic /camera/d405/color/image_raw"
create_runner "$PREVIEW_RUNNER" "OpenCV Collection Preview" "$PREVIEW_CMD"
if [[ "$LAUNCH_REALSENSE_ROS" -eq 1 ]]; then
    create_runner "$CAMERA_RUNNER" "RealSense ROS Cameras" "$CAMERA_CMD"
fi

echo "[collect-launcher] Opening collection tabs..."
if [[ "$LAUNCH_REALSENSE_ROS" -eq 1 ]]; then
    echo "[collect-launcher] 1) RealSense ROS Cameras"
    echo "[collect-launcher] 2) UR3e LeRobot Collector"
else
    echo "[collect-launcher] 1) UR3e LeRobot Collector"
fi
if [[ "$WITH_PREVIEW" -eq 1 ]]; then
    if [[ "$LAUNCH_REALSENSE_ROS" -eq 1 ]]; then
        echo "[collect-launcher] 3) OpenCV Collection Preview"
        gnome-terminal \
            --window --title="RealSense Cameras" --command="$CAMERA_RUNNER" \
            --tab --title="LeRobot Collector" --command="$COLLECTOR_RUNNER" \
            --tab --title="OpenCV Preview" --command="$PREVIEW_RUNNER"
    else
        echo "[collect-launcher] 2) OpenCV Collection Preview"
        gnome-terminal \
            --window --title="LeRobot Collector" --command="$COLLECTOR_RUNNER" \
            --tab --title="OpenCV Preview" --command="$PREVIEW_RUNNER"
    fi
else
    if [[ "$LAUNCH_REALSENSE_ROS" -eq 1 ]]; then
        gnome-terminal \
            --window --title="RealSense Cameras" --command="$CAMERA_RUNNER" \
            --tab --title="LeRobot Collector" --command="$COLLECTOR_RUNNER"
    else
        gnome-terminal \
            --window --title="LeRobot Collector" --command="$COLLECTOR_RUNNER"
    fi
fi

echo "[collect-launcher] Tabs opened."
