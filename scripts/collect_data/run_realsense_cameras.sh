#!/usr/bin/env bash
set -euo pipefail

CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
D455_SERIAL="${D455_SERIAL:-}"
D405_SERIAL="${D405_SERIAL:-}"
D455_PORT="${D455_PORT:-}"
D405_PORT="${D405_PORT:-}"
ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ur3e_rlt_ros_logs}"
LAUNCH_MODE="${LAUNCH_MODE:-separate}"
STARTUP_STAGGER_SEC="${STARTUP_STAGGER_SEC:-2.0}"
SINGLE_CAMERA="${SINGLE_CAMERA:-}"
USE_USB_PORT_ID="${USE_USB_PORT_ID:-0}"

usage() {
    cat <<'EOF'
Usage: scripts/collect_data/run_realsense_cameras.sh [options]

Options:
  --d455-serial SERIAL  Serial number for the front D455.
  --d405-serial SERIAL  Serial number for the wrist D405.
  --d455-port PORT      USB port id for D455, e.g. 2-1. Auto-detected if omitted.
  --d405-port PORT      USB port id for D405, e.g. 2-2. Auto-detected if omitted.
  --width WIDTH         Color width. Default: 640.
  --height HEIGHT       Color height. Default: 480.
  --fps FPS             Color FPS. Default: 30.
  --launch-mode MODE    separate or multi. Default: separate.
  --stagger SEC         Delay between separate camera launches. Default: 2.0.
  --single d455|d405    Launch only one camera for diagnosis.
  --use-usb-port-id     Also pass usb_port_id. Default: serial only.

Environment overrides:
  D455_SERIAL, D405_SERIAL, D455_PORT, D405_PORT, CAMERA_WIDTH,
  CAMERA_HEIGHT, CAMERA_FPS, ROS_LOG_DIR, LAUNCH_MODE, STARTUP_STAGGER_SEC,
  SINGLE_CAMERA, USE_USB_PORT_ID
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --width)
            CAMERA_WIDTH="$2"
            shift 2
            ;;
        --height)
            CAMERA_HEIGHT="$2"
            shift 2
            ;;
        --fps)
            CAMERA_FPS="$2"
            shift 2
            ;;
        --launch-mode)
            LAUNCH_MODE="$2"
            shift 2
            ;;
        --stagger)
            STARTUP_STAGGER_SEC="$2"
            shift 2
            ;;
        --single)
            SINGLE_CAMERA="$2"
            shift 2
            ;;
        --use-usb-port-id)
            USE_USB_PORT_ID=1
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

set +u
source /opt/ros/humble/setup.bash
set -u
mkdir -p "$ROS_LOG_DIR"
export ROS_LOG_DIR

detect_serials() {
    PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import re
import pyrealsense2 as rs

ctx = rs.context()
for dev in ctx.query_devices():
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    try:
        physical_port = dev.get_info(rs.camera_info.physical_port)
    except Exception:
        physical_port = ""
    match = re.search(r"/usb\d+/([^/]+)/", physical_port)
    port = match.group(1) if match else ""
    print(f"{name}|{serial}|{port}")
PY
}

if [[ -z "$D455_SERIAL" || -z "$D405_SERIAL" || -z "$D455_PORT" || -z "$D405_PORT" ]]; then
    while IFS='|' read -r name serial port; do
        if [[ "$name" == *"D455"* && -z "$D455_SERIAL" ]]; then
            D455_SERIAL="$serial"
        fi
        if [[ "$name" == *"D455"* && -z "$D455_PORT" ]]; then
            D455_PORT="$port"
        fi
        if [[ "$name" == *"D405"* && -z "$D405_SERIAL" ]]; then
            D405_SERIAL="$serial"
        fi
        if [[ "$name" == *"D405"* && -z "$D405_PORT" ]]; then
            D405_PORT="$port"
        fi
    done < <(detect_serials)
fi

if [[ -z "$D455_SERIAL" || -z "$D405_SERIAL" || -z "$D455_PORT" || -z "$D405_PORT" ]]; then
    echo "Failed to detect both RealSense cameras." >&2
    echo "D455_SERIAL='$D455_SERIAL' D455_PORT='$D455_PORT' D405_SERIAL='$D405_SERIAL' D405_PORT='$D405_PORT'" >&2
    exit 1
fi

PROFILE="${CAMERA_WIDTH},${CAMERA_HEIGHT},${CAMERA_FPS}"
PIDS=()

cleanup_stale_nodes() {
    echo "[camera] cleanup: stopping stale /camera/d455 and /camera/d405 RealSense nodes..."
    pkill -INT -f "realsense2_camera_node.*__node:=d455.*__ns:=/camera" 2>/dev/null || true
    pkill -INT -f "realsense2_camera_node.*__node:=d405.*__ns:=/camera" 2>/dev/null || true
    pkill -INT -f "ros2 launch realsense2_camera rs_launch.py.*camera_name:=d455" 2>/dev/null || true
    pkill -INT -f "ros2 launch realsense2_camera rs_launch.py.*camera_name:=d405" 2>/dev/null || true
    sleep 1
}

cleanup() {
    trap - INT TERM EXIT
    echo "[camera] cleanup: stopping RealSense ROS process groups..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    # Last-resort cleanup for nodes launched by this script that were orphaned by ros2 launch.
    pkill -INT -f "realsense2_camera_node.*__node:=d455.*__ns:=/camera" 2>/dev/null || true
    pkill -INT -f "realsense2_camera_node.*__node:=d405.*__ns:=/camera" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

cleanup_stale_nodes

launch_single_camera() {
    local name="$1"
    local serial="$2"
    local port="$3"
    local port_args=()
    if [[ "$USE_USB_PORT_ID" -eq 1 ]]; then
        port_args=(usb_port_id:="$port")
        echo "[camera] starting $name serial=$serial usb_port_id=$port profile=$PROFILE"
    else
        echo "[camera] starting $name serial=$serial profile=$PROFILE"
    fi
    setsid ros2 launch realsense2_camera rs_launch.py \
        camera_namespace:=camera \
        camera_name:="$name" \
        serial_no:="_${serial}" \
        "${port_args[@]}" \
        enable_color:=true \
        rgb_camera.color_profile:="$PROFILE" \
        depth_module.color_profile:="$PROFILE" \
        enable_depth:=false \
        enable_infra:=false \
        enable_infra1:=false \
        enable_infra2:=false \
        pointcloud.enable:=false \
        align_depth.enable:=false \
        publish_tf:=false \
        output:=screen &
    PIDS+=("$!")
}

launch_multi_camera() {
    local port_args=()
    if [[ "$USE_USB_PORT_ID" -eq 1 ]]; then
        port_args=(usb_port_id1:="$D455_PORT" usb_port_id2:="$D405_PORT")
        echo "[camera] starting d455 serial=$D455_SERIAL port=$D455_PORT and d405 serial=$D405_SERIAL port=$D405_PORT profile=$PROFILE with rs_multi_camera_launch"
    else
        echo "[camera] starting d455 serial=$D455_SERIAL and d405 serial=$D405_SERIAL profile=$PROFILE with rs_multi_camera_launch"
    fi
    setsid ros2 launch realsense2_camera rs_multi_camera_launch.py \
        camera_namespace1:=camera \
        camera_namespace2:=camera \
        camera_name1:=d455 \
        camera_name2:=d405 \
        serial_no1:="_${D455_SERIAL}" \
        serial_no2:="_${D405_SERIAL}" \
        "${port_args[@]}" \
        enable_color1:=true \
        enable_color2:=true \
        rgb_camera.color_profile1:="$PROFILE" \
        rgb_camera.color_profile2:="$PROFILE" \
        depth_module.color_profile1:="$PROFILE" \
        depth_module.color_profile2:="$PROFILE" \
        enable_depth1:=false \
        enable_depth2:=false \
        enable_infra1:=false \
        enable_infra2:=false \
        enable_infra11:=false \
        enable_infra12:=false \
        enable_infra21:=false \
        enable_infra22:=false \
        pointcloud.enable1:=false \
        pointcloud.enable2:=false \
        align_depth.enable1:=false \
        align_depth.enable2:=false \
        publish_tf1:=false \
        publish_tf2:=false \
        output1:=screen \
        output2:=screen &
    PIDS+=("$!")
}

case "$SINGLE_CAMERA" in
    "")
        ;;
    d455)
        launch_single_camera "d455" "$D455_SERIAL" "$D455_PORT"
        echo "[camera] expected topic:"
        echo "  /camera/d455/color/image_raw"
        echo "[camera] press Ctrl+C to stop the RealSense ROS node."
        wait
        exit $?
        ;;
    d405)
        launch_single_camera "d405" "$D405_SERIAL" "$D405_PORT"
        echo "[camera] expected topic:"
        echo "  /camera/d405/color/image_raw"
        echo "[camera] press Ctrl+C to stop the RealSense ROS node."
        wait
        exit $?
        ;;
    *)
        echo "Unknown --single camera: $SINGLE_CAMERA" >&2
        exit 2
        ;;
esac

case "$LAUNCH_MODE" in
    separate)
        launch_single_camera "d455" "$D455_SERIAL" "$D455_PORT"
        sleep "$STARTUP_STAGGER_SEC"
        launch_single_camera "d405" "$D405_SERIAL" "$D405_PORT"
        ;;
    multi)
        launch_multi_camera
        ;;
    *)
        echo "Unknown --launch-mode: $LAUNCH_MODE" >&2
        exit 2
        ;;
esac

echo "[camera] expected topics:"
echo "  /camera/d455/color/image_raw"
echo "  /camera/d405/color/image_raw"
echo "[camera] press Ctrl+C to stop both RealSense ROS nodes."

wait
