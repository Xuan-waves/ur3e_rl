#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CAMERA_SCRIPT="$PROJECT_ROOT/scripts/collect_data/run_realsense_cameras.sh"
VR_SCRIPT="$PROJECT_ROOT/scripts/hardware/run_ur3e_vr_tabs.sh"
MONITOR_SCRIPT="$PROJECT_ROOT/scripts/rlt_gate/live_rlt_gate_monitor.py"

ROBOT_IP="192.168.5.1"
IMPEDANCE_PROFILE="teleop"
CONDA_ENV_NAME="${UR3E_RLT_CONDA_ENV:-ur3e_rlt}"
CHECKPOINT="$PROJECT_ROOT/outputs/rlt_gate/rlt_gate_20260610_172234/best.pt"
MAX_INFER_HZ="15"
NO_TWIN=1
CAMERA_LAUNCH_MODE="separate"

usage() {
    cat <<'EOF'
Usage: scripts/rlt_gate/run_vr_impedance_gate_test.sh [options]

Starts:
  1) RealSense ROS cameras
  2) live RLT gate monitor window
  3) existing UR3e VR impedance launcher (VR, IK, robot tabs)

Options:
  --robot-ip IP            UR controller IP. Default: 192.168.5.1
  --impedance-profile P    Impedance profile. Default: teleop
  --checkpoint PATH        RLT gate checkpoint. Default: latest both+resnet18 test checkpoint
  --max-infer-hz HZ        Gate inference cap. Default: 15
  --conda-env NAME         Conda env. Default: $UR3E_RLT_CONDA_ENV or ur3e_rlt
  --with-twin              Start MuJoCo twin in IK tab.
  --camera-launch-mode M   RealSense launch mode: separate or multi. Default: separate
  -h, --help               Show help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)
            ROBOT_IP="${2:?--robot-ip requires a value}"
            shift 2
            ;;
        --impedance-profile)
            IMPEDANCE_PROFILE="${2:?--impedance-profile requires a value}"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="${2:?--checkpoint requires a value}"
            shift 2
            ;;
        --max-infer-hz)
            MAX_INFER_HZ="${2:?--max-infer-hz requires a value}"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="${2:?--conda-env requires a value}"
            shift 2
            ;;
        --with-twin)
            NO_TWIN=0
            shift
            ;;
        --camera-launch-mode)
            CAMERA_LAUNCH_MODE="${2:?--camera-launch-mode requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "[rlt-test] gnome-terminal not found." >&2
    exit 1
fi

quote() {
    printf "%q" "$1"
}

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
TMP_DIR="/tmp/ur3e_rlt_gate_test_${RUN_ID}"
mkdir -p "$TMP_DIR"

create_common_runner() {
    local path="$1"
    local title="$2"
    local command_body="$3"

    cat > "$path" <<EOF
#!/usr/bin/env bash
set +e
PROJECT_ROOT=$(quote "$PROJECT_ROOT")
CONDA_ENV_NAME=$(quote "$CONDA_ENV_NAME")
cd "\${PROJECT_ROOT}" || exit 1
if [ -f /opt/ros/humble/setup.bash ]; then
    set +u
    . /opt/ros/humble/setup.bash
    set -u
fi
if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)" || true
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "\$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "\$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if [ -n "\${CONDA_ENV_NAME}" ]; then
    conda activate "\${CONDA_ENV_NAME}"
fi
echo "== $title"
echo "working dir: \${PROJECT_ROOT}"
echo "conda env  : \${CONDA_ENV_NAME}"
echo
$command_body
status=\$?
echo
echo "[rlt-test] $title exited with code \${status}"
echo "[rlt-test] Press Ctrl+D or type exit to close this tab."
exec bash
EOF
    chmod +x "$path"
}

CAMERA_RUNNER="$TMP_DIR/cameras.sh"
MONITOR_RUNNER="$TMP_DIR/gate_monitor.sh"

create_common_runner "$CAMERA_RUNNER" "RealSense ROS Cameras" \
    "$(quote "$CAMERA_SCRIPT") --launch-mode $(quote "$CAMERA_LAUNCH_MODE")"

create_common_runner "$MONITOR_RUNNER" "RLT Gate Live Monitor" \
    "python -u $(quote "$MONITOR_SCRIPT") --checkpoint $(quote "$CHECKPOINT") --max-infer-hz $(quote "$MAX_INFER_HZ")"

echo "[rlt-test] Opening RealSense + RLT gate monitor tabs..."
gnome-terminal \
    --window --title="RLT Gate Test Cameras" --command="$CAMERA_RUNNER" \
    --tab --title="RLT Gate Live" --command="$MONITOR_RUNNER"

sleep 1

VR_ARGS=(--robot-ip "$ROBOT_IP" --control-mode impedance --impedance-profile "$IMPEDANCE_PROFILE" --conda-env "$CONDA_ENV_NAME")
if [[ "$NO_TWIN" -eq 1 ]]; then
    VR_ARGS+=(--no-twin)
fi

echo "[rlt-test] Opening UR3e VR impedance tabs..."
"$VR_SCRIPT" "${VR_ARGS[@]}"

echo "[rlt-test] Launched."
