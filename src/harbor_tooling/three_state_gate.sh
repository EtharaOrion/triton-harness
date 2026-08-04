#!/usr/bin/env bash
# Three-state gate for a carved Harbor task image, run with NO network.
#
#   RED    -- the image exactly as shipped (carve in place)  -> reward 0
#   GREEN  -- the carved files restored from the HOST oracle -> reward 1
#
# The oracle is byte-identical to repos-src/<repo> (build.sh asserts this), so
# the GREEN leg is simultaneously the "intact repo still passes" check and the
# "solution/ still works now that it lives outside the image" check. That
# second property is the whole point of this script: the solution used to be
# COPYed into the image, and removing it must not break the oracle path.
#
# usage: three_state_gate.sh <image> <repo-path-in-image> <host-oracle-dir> <verifier-cmd> [host-verifier-script]
set -uo pipefail

IMG=$1
REPO=$2
ORACLE=$3
VERIFIER=$4
HOST_TEST=${5:-}

NAME="gate-$(echo "$IMG" | tr ':/' '--')-$$"
RC=0

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "############################################################"
echo "# three-state gate: $IMG"
echo "############################################################"

docker run -d --name "$NAME" --network=none --entrypoint sleep "$IMG" infinity >/dev/null

# Harbor mounts the verifier for tracks whose task.toml declares no
# [verifier].command; mirror that here rather than assuming it is baked in.
if [ -n "$HOST_TEST" ]; then
    docker cp "$HOST_TEST" "$NAME:/opt/harbor/test.sh"
    docker exec "$NAME" chmod 0555 /opt/harbor/test.sh
fi

run_leg() {
    local label=$1 want=$2
    echo
    echo "--- $label (expect reward=$want) ---"
    docker exec "$NAME" sh -c 'rm -rf /logs/verifier; mkdir -p /logs/verifier'
    docker exec "$NAME" sh -c "$VERIFIER" > "/tmp/$NAME.$label.log" 2>&1
    local got
    got=$(docker exec "$NAME" sh -c 'cat /logs/verifier/reward.txt 2>/dev/null' | tr -d '[:space:]')
    echo "verifier tail:"
    tail -6 "/tmp/$NAME.$label.log" | sed 's/^/    /'
    echo "reward.txt = '${got:-<missing>}'"
    case "$got" in
        "$want"|"$want.0"|"$want.00") echo "PASS: $label -> $got" ;;
        *) echo "FAIL: $label -> '${got:-<missing>}', wanted $want"; RC=1 ;;
    esac
}

run_leg RED 0

echo
echo "--- restoring oracle from HOST (docker cp), i.e. from OUTSIDE the image ---"
docker exec "$NAME" sh -c "ls -d $ORACLE 2>/dev/null && echo 'ORACLE ALREADY IN IMAGE -- LEAK' && exit 1; true"
docker cp "$ORACLE/." "$NAME:$REPO/"
echo "restored $(find "$ORACLE" -type f | wc -l | tr -d ' ') files into $REPO"

run_leg GREEN 1

echo
if [ "$RC" -eq 0 ]; then
    echo "GATE PASS: $IMG"
else
    echo "GATE FAIL: $IMG"
fi
exit "$RC"
