#!/usr/bin/env bash
# Manage the Blurrinator runtime: Django web + django-q worker.
# Usage: ./app.sh {start|stop|restart|status|logs}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/venv/bin/python"
PID_DIR="$ROOT/.run"
LOG_DIR="$ROOT/.run/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

WEB_PIDFILE="$PID_DIR/web.pid"
WORKER_PIDFILE="$PID_DIR/worker.pid"
HOST="${BLURRINATOR_HOST:-0.0.0.0}"
PORT="${BLURRINATOR_PORT:-8013}"

is_running() {
    local pidfile="$1"
    [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

start_one() {
    local name="$1" pidfile="$2"; shift 2
    if is_running "$pidfile"; then
        echo "$name already running (pid $(cat "$pidfile"))"
        return 0
    fi
    echo "Starting $name…"
    cd "$ROOT"
    nohup "$@" >>"$LOG_DIR/$name.log" 2>&1 &
    echo $! > "$pidfile"
    sleep 0.6
    if is_running "$pidfile"; then
        echo "  $name pid $(cat "$pidfile") — log: $LOG_DIR/$name.log"
    else
        echo "  $name FAILED to start; tail -50 $LOG_DIR/$name.log:"
        tail -50 "$LOG_DIR/$name.log" || true
        rm -f "$pidfile"
        return 1
    fi
}

stop_one() {
    local name="$1" pidfile="$2"
    if ! is_running "$pidfile"; then
        echo "$name not running"
        rm -f "$pidfile"
        return 0
    fi
    local pid; pid="$(cat "$pidfile")"
    echo "Stopping $name (pid $pid)…"
    # qcluster forks workers; kill the whole process group.
    local pgid; pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "$pgid" ]]; then
        kill -TERM "-$pgid" 2>/dev/null || true
    else
        kill -TERM "$pid" 2>/dev/null || true
    fi
    for _ in {1..40}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "  TERM didn't take, sending KILL"
        [[ -n "$pgid" ]] && kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
}

status_one() {
    local name="$1" pidfile="$2"
    if is_running "$pidfile"; then
        echo "  $name: running (pid $(cat "$pidfile"))"
    else
        echo "  $name: stopped"
    fi
}

case "${1:-status}" in
    start)
        start_one web "$WEB_PIDFILE" "$PY" manage.py runserver --noreload "$HOST:$PORT"
        start_one worker "$WORKER_PIDFILE" "$PY" manage.py qcluster
        echo "Open: http://$HOST:$PORT/"
        ;;
    stop)
        stop_one worker "$WORKER_PIDFILE"
        stop_one web "$WEB_PIDFILE"
        ;;
    restart)
        "$0" stop
        sleep 0.5
        "$0" start
        ;;
    restart-worker)
        # For when only services.py / model code changed; web doesn't need a kick.
        stop_one worker "$WORKER_PIDFILE"
        sleep 0.3
        start_one worker "$WORKER_PIDFILE" "$PY" manage.py qcluster
        ;;
    status)
        echo "Blurrinator:"
        status_one web "$WEB_PIDFILE"
        status_one worker "$WORKER_PIDFILE"
        ;;
    logs)
        exec tail -F "$LOG_DIR/web.log" "$LOG_DIR/worker.log"
        ;;
    *)
        echo "usage: $0 {start|stop|restart|restart-worker|status|logs}" >&2
        exit 1
        ;;
esac
