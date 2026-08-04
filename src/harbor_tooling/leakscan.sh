#!/usr/bin/env bash
# Content-based answer-leak scan of a whole container filesystem.
#
# Reads distinctive carved lines (one per line) from $TRIPWIRE_FILE -- lines
# that occur in the deleted subsystem and in NO surviving file of the repo --
# and greps every regular file on the filesystem for them, fixed-string and
# binary-as-text. That catches a leak by ANY route: a renamed directory, a
# tarball member, a loose git object, a .pyc docstring, a string baked into an
# ELF, an editor backup. Path-based checks catch none of those.
#
# Used twice: as a build-time gate inside each Dockerfile (tripwires arrive on
# a BuildKit bind mount, which leaves no layer), and as the post-build
# acceptance proof.
#
# Exit 0 = clean, 1 = leak, 2 = the check could not run. Fails closed.
set -uo pipefail

TRIPWIRE_FILE="${TRIPWIRE_FILE:-/tmp/.harbor-tripwires}"
LABEL="${LEAKSCAN_LABEL:-image}"

if [ ! -s "${TRIPWIRE_FILE}" ]; then
    echo "LEAKSCAN FATAL: no tripwires at ${TRIPWIRE_FILE}" >&2
    exit 2
fi

# Copy the patterns onto the container filesystem. The bind-mounted original is
# re-read by every one of the parallel grep workers, and on a VM-backed mount
# that alone costs minutes.
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cp "${TRIPWIRE_FILE}" "${work}/tw"
ntw=$(grep -c . < "${work}/tw")
if [ "${ntw}" -lt 1 ]; then
    echo "LEAKSCAN FATAL: tripwire file is empty" >&2
    exit 2
fi

# /proc, /sys and /dev are kernel views, not image content (and /proc/kcore is
# a multi-terabyte sparse file that would stall the scan forever). The scratch
# dir and the bind mount hold the tripwires themselves, so they are excluded.
find / \
    -path /proc -prune -o \
    -path /sys -prune -o \
    -path /dev -prune -o \
    -path "${work}" -prune -o \
    -path "${TRIPWIRE_FILE}" -prune -o \
    -type f -print 2>/dev/null > "${work}/files"

nfiles=$(wc -l < "${work}/files" | tr -d ' ')
echo "leakscan[${LABEL}]: ${ntw} tripwire lines vs ${nfiles} files"

LC_ALL=C tr '\n' '\0' < "${work}/files" \
  | xargs -0 -r -P "$(nproc)" -n 300 \
      grep -l -F -a --binary-files=text -f "${work}/tw" 2>/dev/null \
  | sort -u > "${work}/hits"

# xargs -P cannot report grep's exit status usefully (123 for "no match in some
# batch"), so the verdict is taken from the hit list, which is unambiguous.
if [ -s "${work}/hits" ]; then
    echo "LEAKSCAN FAIL [${LABEL}]: carved source content found inside the image" >&2
    while read -r f; do
        echo "  HIT ${f}" >&2
        LC_ALL=C grep -n -F -a --binary-files=text -f "${work}/tw" "${f}" 2>/dev/null \
            | head -2 | cut -c1-140 | sed 's/^/        /' >&2
    done < "${work}/hits"
    exit 1
fi

echo "LEAKSCAN PASS [${LABEL}]: 0 hits for ${ntw} carved tripwire lines across ${nfiles} files"
exit 0
