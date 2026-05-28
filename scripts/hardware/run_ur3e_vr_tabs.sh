#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MAIN_SCRIPT="$PROJECT_ROOT/scripts/hardware/ur3e_vr_servoj_ros2.py"

ROBOT_IP="192.168.5.1"
XML_PATH=""
DRY_RUN=0
NO_TWIN=0
CONTROL_MODE="impedance"
IMPEDANCE_PROFILE="teleop"
CONDA_ENV_NAME="${UR3E_VR_CONDA_ENV:-vr}"

usage() {
    cat <<'EOF'
Usage: scripts/hardware/run_ur3e_vr_tabs.sh [options]

Options:
  --dry-run              Run robot tab without connecting to UR controller.
  --robot-ip IP          UR controller IP. Default: 192.168.5.1
  --xml PATH             MuJoCo XML path for IK and digital twin.
  --no-twin              Do not start MuJoCo viewer inside the IK tab.
  --control-mode MODE    Robot control mode: impedance or servoj. Default: impedance.
  --impedance-profile P  Impedance profile from real_teleop/impedance/config.py. Default: teleop.
  --conda-env NAME       Conda environment to activate. Default: $UR3E_VR_CONDA_ENV or vr.
  --no-conda             Use current shell Python instead of activating conda.
  -h, --help             Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --robot-ip)
            ROBOT_IP="${2:?--robot-ip requires a value}"
            shift 2
            ;;
        --xml)
            XML_PATH="${2:?--xml requires a value}"
            shift 2
            ;;
        --no-twin)
            NO_TWIN=1
            shift
            ;;
        --control-mode)
            CONTROL_MODE="${2:?--control-mode requires a value}"
            if [ "$CONTROL_MODE" != "impedance" ] && [ "$CONTROL_MODE" != "servoj" ]; then
                echo "[launcher] --control-mode must be impedance or servoj" >&2
                exit 2
            fi
            shift 2
            ;;
        --impedance-profile)
            IMPEDANCE_PROFILE="${2:?--impedance-profile requires a value}"
            shift 2
            ;;
        --conda-env)
            CONDA_ENV_NAME="${2:?--conda-env requires a value}"
            shift 2
            ;;
        --no-conda)
            CONDA_ENV_NAME=""
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[launcher] Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "[launcher] Missing script: $MAIN_SCRIPT" >&2
    exit 1
fi

if ! command -v gnome-terminal >/dev/null 2>&1; then
    echo "[launcher] gnome-terminal not found. Install it or run node commands manually." >&2
    exit 1
fi

quote() {
    printf "%q" "$1"
}

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
PID_DIR="/tmp/ur3e_vr_tabs_${RUN_ID}"
mkdir -p "$PID_DIR"

create_runner() {
    local runner_path="$1"
    local title="$2"
    local node_name="$3"
    local pid_name="$4"

    cat > "$runner_path" <<EOF
#!/usr/bin/env bash
set +e
set -m

PROJECT_ROOT=$(quote "$PROJECT_ROOT")
MAIN_SCRIPT=$(quote "$MAIN_SCRIPT")
PID_DIR=$(quote "$PID_DIR")
PID_FILE="\${PID_DIR}/$pid_name.pid"
TITLE=$(quote "$title")
NODE_NAME=$(quote "$node_name")
ROBOT_IP=$(quote "$ROBOT_IP")
XML_PATH=$(quote "$XML_PATH")
DRY_RUN=$DRY_RUN
NO_TWIN=$NO_TWIN
CONTROL_MODE=$(quote "$CONTROL_MODE")
IMPEDANCE_PROFILE=$(quote "$IMPEDANCE_PROFILE")
CONDA_ENV_NAME=$(quote "$CONDA_ENV_NAME")

mkdir -p "\${PID_DIR}"
cd "\${PROJECT_ROOT}" || exit 1
if [ -f /opt/ros/humble/setup.bash ]; then
    . /opt/ros/humble/setup.bash
fi
export ROS_LOG_DIR="\${ROS_LOG_DIR:-/tmp/ros_logs}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "\${ROS_LOG_DIR}"

kill_all_ur3e_tabs() {
    echo
    echo "[launcher] Ctrl+C received. Stopping all UR3e VR tab processes..."
    for pid_file in "\${PID_DIR}"/*.pid; do
        [ -f "\${pid_file}" ] || continue
        pid="\$(cat "\${pid_file}" 2>/dev/null || true)"
        if [ -n "\${pid}" ] && kill -0 "\${pid}" 2>/dev/null; then
            echo "[launcher] sending SIGINT to PID \${pid}"
            kill -INT "\${pid}" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid_file in "\${PID_DIR}"/*.pid; do
        [ -f "\${pid_file}" ] || continue
        pid="\$(cat "\${pid_file}" 2>/dev/null || true)"
        if [ -n "\${pid}" ] && kill -0 "\${pid}" 2>/dev/null; then
            echo "[launcher] sending SIGTERM to PID \${pid}"
            kill -TERM "\${pid}" 2>/dev/null || true
        fi
    done
}

cleanup_this_tab() {
    rm -f "\${PID_FILE}"
}

activate_conda_env() {
    local env_name="\$1"
    [ -n "\${env_name}" ] || return 0

    if command -v conda >/dev/null 2>&1; then
        eval "\$(conda shell.bash hook)" || true
    elif [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "\$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "[launcher] Could not find conda. Use --no-conda or check your installation." >&2
        return 1
    fi

    conda activate "\${env_name}"
}

build_args() {
    ARGS=(--node "\${NODE_NAME}" --robot-ip "\${ROBOT_IP}" --control-mode "\${CONTROL_MODE}" --impedance-profile "\${IMPEDANCE_PROFILE}")
    if [ -n "\${XML_PATH}" ]; then
        ARGS+=(--xml "\${XML_PATH}")
    fi
    if [ "\${NODE_NAME}" = "robot" ] && [ "\${DRY_RUN}" -eq 1 ]; then
        ARGS+=(--dry-run)
    fi
    if [ "\${NODE_NAME}" = "ik" ] && [ "\${NO_TWIN}" -eq 1 ]; then
        ARGS+=(--no-twin)
    fi
}

check_python_ros() {
    python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"ROS Humble rclpy requires Python 3.10 here, but python3 is {sys.version.split()[0]} at {sys.executable}"
    )
import rclpy
PY
}

trap "kill_all_ur3e_tabs; cleanup_this_tab; exit 130" INT TERM
trap "cleanup_this_tab" EXIT

echo "== \${TITLE}"
echo "working dir : \${PROJECT_ROOT}"
echo "script      : \${MAIN_SCRIPT}"
echo "node        : \${NODE_NAME}"
echo "robot ip    : \${ROBOT_IP}"
echo "control mode: \${CONTROL_MODE}"
echo "imp profile : \${IMPEDANCE_PROFILE}"
echo "conda env   : \${CONDA_ENV_NAME:-<current>}"
echo "ros logs    : \${ROS_LOG_DIR}"
echo

if ! activate_conda_env "\${CONDA_ENV_NAME}"; then
    echo "[launcher] Failed to activate conda environment: \${CONDA_ENV_NAME}" >&2
    echo "[launcher] Press Ctrl+D or type exit to close this tab."
    exec bash
fi

if ! check_python_ros; then
    echo
    echo "[launcher] Python/ROS environment check failed."
    echo "[launcher] Use a Python 3.10 conda env, or run with --no-conda if /usr/bin/python3 has the needed deps."
    echo "[launcher] Press Ctrl+D or type exit to close this tab."
    exec bash
fi

build_args
(
    python3 -u "\${MAIN_SCRIPT}" "\${ARGS[@]}"
) &
python_pid=\$!
echo "\${python_pid}" > "\${PID_FILE}"
fg "%1"
exit_code=\$?
if [ "\${exit_code}" -eq 130 ]; then
    kill_all_ur3e_tabs
fi

echo
echo "[launcher] \${TITLE} exited with code \${exit_code}"
echo "[launcher] Press Ctrl+D or type exit to close this tab."
exec bash
EOF

    chmod +x "$runner_path"
}

VR_RUNNER="$PID_DIR/vr_runner.sh"
IK_RUNNER="$PID_DIR/ik_runner.sh"
ROBOT_RUNNER="$PID_DIR/robot_runner.sh"

create_runner "$VR_RUNNER" "UR3e VR Input" "vr" "vr"
create_runner "$IK_RUNNER" "UR3e IK + MuJoCo Twin" "ik" "ik"
create_runner "$ROBOT_RUNNER" "UR3e Robot" "robot" "robot"

echo "[launcher] Opening one GNOME Terminal window with three tabs..."
echo "[launcher] 1) VR Input          env=${CONDA_ENV_NAME:-<current>}"
echo "[launcher] 2) IK + MuJoCo Twin  env=${CONDA_ENV_NAME:-<current>}"
echo "[launcher] 3) Robot             env=${CONDA_ENV_NAME:-<current>}"
echo "[launcher] dry_run=$DRY_RUN no_twin=$NO_TWIN robot_ip=$ROBOT_IP control_mode=$CONTROL_MODE impedance_profile=$IMPEDANCE_PROFILE"
echo "[launcher] Shared PID directory: $PID_DIR"

gnome-terminal \
    --window --title="UR3e VR" --command="$VR_RUNNER" \
    --tab --title="UR3e IK + Twin" --command="$IK_RUNNER" \
    --tab --title="UR3e Robot" --command="$ROBOT_RUNNER"

echo "[launcher] Tabs opened."
