#!/usr/bin/env bash
# Point `data/` at the corpora, wherever they live on this machine.
#
#   ./link_datasets.sh \
#       --forest-persons     /path/to/forest_persons \
#       --forest-belongings  /path/to/ForestBelongings \
#       --coco               /path/to/COCO
#
# Symlinks rather than copies: these corpora run to tens of thousands of frames
# and nothing here needs its own copy. Any subset may be given; the rest stay
# unlinked and the checks report them as absent.
#
# The alternative to this script is `UC_DATA_ROOT`, or editing
# configs/paths.yaml. Use whichever suits; the code reads all three the same way.
#
# Every link is verified after it is made. A symlink whose target does not exist
# is created happily by `ln` and then reports itself as a perfectly ordinary
# directory to anything that does not look twice -- and an evaluation over a
# dangling link does not crash, it scores zero and says nothing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${REPO_ROOT}/data"

FOREST_PERSONS=""; FOREST_BELONGINGS=""; COCO=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --forest-persons)    FOREST_PERSONS="$2"; shift 2 ;;
        --forest-belongings) FOREST_BELONGINGS="$2"; shift 2 ;;
        --coco)              COCO="$2"; shift 2 ;;
        -h|--help)           sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${DATA_DIR}"
FAILED=0

link() {  # name, target
    local name="$1" target="$2" link_path="${DATA_DIR}/$1"
    [ -z "${target}" ] && return 0
    if [ ! -d "${target}" ]; then
        echo "REFUSED ${name}: ${target} is not a directory" >&2
        FAILED=1; return 0
    fi
    # Resolve before linking: a relative target would be interpreted relative to
    # data/, not to where the caller typed it, and would land somewhere neither
    # of us meant.
    target="$(cd "${target}" && pwd)"
    [ -L "${link_path}" ] && rm "${link_path}"
    if [ -e "${link_path}" ]; then
        echo "REFUSED ${name}: ${link_path} exists and is not a symlink; not touching it" >&2
        FAILED=1; return 0
    fi
    ln -s "${target}" "${link_path}"
    echo "linked  ${name} -> ${target}"
}

# Counting, not just existence. A corpus that unpacked partially, or a link into
# a directory that happens to have the right name, both pass an -e test and fail
# the run much later.
check() {  # name, expected-subdirectory
    local name="$1" sub="$2" link_path="${DATA_DIR}/$1"
    [ -L "${link_path}" ] || return 0
    if [ ! -d "${link_path}/${sub}" ]; then
        echo "  WARNING ${name}: no ${sub}/ under the target — wrong directory?" >&2
        FAILED=1; return 0
    fi
    # `|| true` on the find, not on the pipeline: `head` closing the pipe sends
    # SIGPIPE to find, which under `set -o pipefail` fails the whole pipeline and
    # under `set -e` ends the script -- silently, mid-verification, with the
    # remaining datasets never checked. A verifier that stops verifying and
    # returns success is worse than no verifier.
    local n
    n="$({ find -L "${link_path}/${sub}" -maxdepth 2 -type f 2>/dev/null || true; } \
         | head -20001 | wc -l)"
    if [ "${n}" -eq 0 ]; then
        echo "  WARNING ${name}: ${sub}/ is empty" >&2
        FAILED=1
    else
        [ "${n}" -gt 20000 ] && n=">20000"
        echo "  ${name}: ${n} file(s) under ${sub}/"
    fi
}

link forest_persons    "${FOREST_PERSONS}"
link forest_belongings "${FOREST_BELONGINGS}"
link coco              "${COCO}"

echo
echo "verifying:"
check forest_persons    annotations
check forest_belongings annotations
check coco              annotations

echo
if [ "${FAILED}" -ne 0 ]; then
    echo "one or more datasets did not verify. See DATA.md for what each should contain." >&2
    exit 1
fi
echo "all linked datasets verified. Next: python prepare.py"
