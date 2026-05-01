#!/usr/bin/env bash
set -u

PRESERVE_ROOTS="3732147 3736881"
PRESERVE_SET=" $PRESERVE_ROOTS "
for root in $PRESERVE_ROOTS; do
    for child in $(pgrep -P "$root" 2>/dev/null); do
        PRESERVE_SET="$PRESERVE_SET$child "
    done
done
echo "Preserving PIDs:$PRESERVE_SET"

KILL_PIDS=""
for pid in $(pgrep -f python 2>/dev/null); do
    cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
    case "$cwd" in
        *LabelMix*) ;;
        *) continue ;;
    esac
    case "$PRESERVE_SET" in
        *" $pid "*) continue ;;
    esac
    KILL_PIDS="$KILL_PIDS $pid"
done

count=$(echo $KILL_PIDS | wc -w)
echo "Will kill $count PIDs"
if [ "$count" -eq 0 ]; then
    exit 0
fi

echo "--- SIGTERM ---"
echo $KILL_PIDS | xargs -r kill -TERM 2>/dev/null
sleep 3

remaining=""
for pid in $KILL_PIDS; do
    [ -d "/proc/$pid" ] && remaining="$remaining $pid"
done
rcount=$(echo $remaining | wc -w)
echo "After SIGTERM still alive: $rcount"
if [ "$rcount" -gt 0 ]; then
    echo "--- SIGKILL ---"
    echo $remaining | xargs -r kill -KILL 2>/dev/null
    sleep 2
fi

final=0
for pid in $KILL_PIDS; do
    [ -d "/proc/$pid" ] && final=$((final+1))
done
echo "Final still alive: $final"
