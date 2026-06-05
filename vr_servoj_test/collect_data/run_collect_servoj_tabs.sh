#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${UR3E_RLT_CONDA_ENV:-ur3e_rlt}"

config_value() {
    local attr="$1"
    (cd "$REPO_ROOT" && python3 -c "from vr_servoj_test.collect_data.config import VrServoJCollectConfig; print(getattr(VrServoJCollectConfig, '$attr'))")
}

DATASET_ROOT="$(config_value dataset_root)"
DATASET_NAME="$(config_value dataset_name)"
TASK="$(config_value task)"
STATE_MODE="$(config_value state_mode)"
ACTION_MODE="$(config_value action_mode)"
EE_ACTION_POSITION_MODE="$(config_value ee_action_position_mode)"
COMMANDED_JOINT_TARGET_TOPIC="$(config_value commanded_joint_target_topic)"
STATUS_PANEL=1
STATUS_HZ="$(config_value status_hz)"
MAX_EPISODES="$(config_value max_episodes)"
MAX_DT_FRONT_IMAGE="$(config_value max_dt_front_image)"
MAX_DT_WRIST_IMAGE="$(config_value max_dt_wrist_image)"
MAX_DT_STATE="$(config_value max_dt_state)"
MAX_DT_ACTION="$(config_value max_dt_action)"
SYNC_REFERENCE="$(config_value sync_reference)"
WITH_PREVIEW=1

usage() {
    cat <<'EOF'
Usage: vr_servoj_test/collect_data/run_collect_servoj_tabs.sh [options]

Collect LeRobot data from the pure VR servoJ teleop path.

Start prerequisites separately:
  scripts/hardware/run_ur3e_vr_tabs.sh --robot-ip 192.168.5.1 --control-mode servoj --no-twin --conda-env ur3e_rlt
  scripts/collect_data/run_realsense_cameras.sh

Options:
  --conda-env NAME       Conda env. Default: $UR3E_RLT_CONDA_ENV or ur3e_rlt.
  --dataset-root PATH    Dataset parent directory.
  --dataset-name NAME    Dataset session prefix.
  --task TEXT            Task text saved to LeRobot.
  --max-episodes N       Stop after saving N episodes. 0 means unlimited.
  --state-mode MODE      jointspace or eepose. Default: eepose.
  --action-mode MODE     jointspace or eepose. Default: jointspace.
  --ee-action-position-mode relative|absolute
                        Only used when --action-mode eepose. Default: relative.
  --commanded-joint-target-topic TOPIC
                        Robot-node commanded servoJ target topic.
  --max-dt-front SEC     D455/front nearest-image tolerance.
  --max-dt-wrist SEC     D405/wrist nearest-image tolerance.
  --max-dt-state SEC     Robot state tolerance.
  --max-dt-action SEC    Joint/EE action and VR command tolerance.
  --sync-reference MODE  front, wrist, or timer. Default: front.
  --no-preview           Do not open the OpenCV preview tab.
  --no-status-panel      Disable the in-place collector status panel.
  --status-hz HZ         Status panel refresh rate.
EOF
}

quote() {
    printf "%q" "$1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV_NAME="$2"; shift 2 ;;
        --dataset-root)
            DATASET_ROOT="$2"; shift 2 ;;
        --dataset-name)
            DATASET_NAME="$2"; shift 2 ;;
        --task)
            TASK="$2"; shift 2 ;;
        --max-episodes)
            MAX_EPISODES="$2"; shift 2 ;;
        --state-mode)
            STATE_MODE="$2"; shift 2 ;;
        --action-mode)
            ACTION_MODE="$2"; shift 2 ;;
        --ee-action-position-mode)
            EE_ACTION_POSITION_MODE="$2"; shift 2 ;;
        --commanded-joint-target-topic)
            COMMANDED_JOINT_TARGET_TOPIC="$2"; shift 2 ;;
        --max-dt-front)
            MAX_DT_FRONT_IMAGE="$2"; shift 2 ;;
        --max-dt-wrist)
            MAX_DT_WRIST_IMAGE="$2"; shift 2 ;;
        --max-dt-state)
            MAX_DT_STATE="$2"; shift 2 ;;
        --max-dt-action)
            MAX_DT_ACTION="$2"; shift 2 ;;
        --sync-reference)
            SYNC_REFERENCE="$2"; shift 2 ;;
        --no-preview)
            WITH_PREVIEW=0; shift ;;
        --no-status-panel)
            STATUS_PANEL=0; shift ;;
        --status-hz)
            STATUS_HZ="$2"; shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "[servoj-collect] gnome-terminal not found." >&2
    exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PID_DIR="/tmp/ur3e_servoj_collect_tabs_${RUN_ID}"
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
    elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "[servoj-collect] Could not find conda." >&2
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
    echo "[servoj-collect] Failed to activate conda environment: \${CONDA_ENV_NAME}" >&2
    exec bash
fi

$body
exit_code=\$?
echo
echo "[servoj-collect] \${TITLE} exited with code \${exit_code}"
echo "[servoj-collect] Press Ctrl+D or type exit to close this tab."
exec bash
EOF
    chmod +x "$runner_path"
}

COLLECTOR_CMD="set +u; source /opt/ros/humble/setup.bash; set -u; python vr_servoj_test/collect_data/collect_ur3e_vr_servoj.py --dataset-root $(quote "$DATASET_ROOT") --dataset-name $(quote "$DATASET_NAME") --task $(quote "$TASK") --state-mode $(quote "$STATE_MODE") --action-mode $(quote "$ACTION_MODE") --ee-action-position-mode $(quote "$EE_ACTION_POSITION_MODE") --commanded-joint-target-topic $(quote "$COMMANDED_JOINT_TARGET_TOPIC") --max-dt-front-image $(quote "$MAX_DT_FRONT_IMAGE") --max-dt-wrist-image $(quote "$MAX_DT_WRIST_IMAGE") --max-dt-state $(quote "$MAX_DT_STATE") --max-dt-action $(quote "$MAX_DT_ACTION") --sync-reference $(quote "$SYNC_REFERENCE") --max-episodes $(quote "$MAX_EPISODES") --status-hz $(quote "$STATUS_HZ")"
if [[ "$STATUS_PANEL" -eq 1 ]]; then
    COLLECTOR_CMD+=" --status-panel"
else
    COLLECTOR_CMD+=" --no-status-panel"
fi

PREVIEW_CMD="set +u; source /opt/ros/humble/setup.bash; set -u; python scripts/collect_data/preview_collection_topic.py --front-topic /camera/d455/color/image_raw --wrist-topic /camera/d405/color/image_raw"

COLLECTOR_RUNNER="$PID_DIR/collector.sh"
PREVIEW_RUNNER="$PID_DIR/preview.sh"
create_runner "$COLLECTOR_RUNNER" "UR3e VR ServoJ Collector" "$COLLECTOR_CMD"
create_runner "$PREVIEW_RUNNER" "OpenCV Collection Preview" "$PREVIEW_CMD"

echo "[servoj-collect] Opening tabs..."
if [[ "$WITH_PREVIEW" -eq 1 ]]; then
    gnome-terminal \
        --window --title="VR ServoJ Collector" --command="$COLLECTOR_RUNNER" \
        --tab --title="OpenCV Preview" --command="$PREVIEW_RUNNER"
else
    gnome-terminal --window --title="VR ServoJ Collector" --command="$COLLECTOR_RUNNER"
fi
echo "[servoj-collect] Tabs opened."
