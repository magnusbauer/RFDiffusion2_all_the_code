#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_OUTPUT_DIR="${REPO_DIR}/rf_diffusion/model_weights"

OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
FORCE=0
DRY_RUN=0

WEIGHT_NAMES=(
    RFD_173.pt
    RFD_140.pt
    RFD_45.pt
    ppi_robust_struct.pt
)

WEIGHT_URLS=(
    https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_173.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2/model_weights/RFD_140.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2-mi/model_weights/RFD_45.pt
    https://files.ipd.uw.edu/pub/rfdiffusion2-mi/model_weights/ppi_robust_struct.pt
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Download all published RFdiffusion2 and RFdiffusion2-MI model weights.

Options:
  --output-dir DIR  Download into DIR.
                    Defaults to ${DEFAULT_OUTPUT_DIR}.
  --force           Replace existing completed weight files.
  --dry-run         Print the downloads without starting them.
  -h, --help        Show this help message.

Files:
  RFdiffusion2:     RFD_173.pt, RFD_140.pt
  Regression tests: RFD_45.pt
  RFdiffusion2-MI:  ppi_robust_struct.pt

Interrupted downloads are kept as NAME.partial and resumed on the next run.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output-dir)
            [[ "$#" -ge 2 && -n "$2" ]] || die "--output-dir requires a directory"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#--output-dir=}"
            [[ -n "${OUTPUT_DIR}" ]] || die "--output-dir requires a directory"
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
    else
        die "curl or wget is required"
    fi

    mkdir -p -- "${OUTPUT_DIR}"
fi

for index in "${!WEIGHT_NAMES[@]}"; do
    name="${WEIGHT_NAMES[index]}"
    url="${WEIGHT_URLS[index]}"
    output="${OUTPUT_DIR}/${name}"
    partial="${output}.partial"

    if [[ -f "${output}" && "${FORCE}" -eq 0 ]]; then
        printf 'Already present, skipping: %s\n' "${output}"
        continue
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '%s -> %s\n' "${url}" "${output}"
        continue
    fi

    if [[ "${FORCE}" -eq 1 ]]; then
        rm -f -- "${partial}"
    fi

    printf 'Downloading %s\n' "${name}"
    if [[ "${DOWNLOADER}" == curl ]]; then
        curl -fL --retry 3 --retry-delay 5 --continue-at - \
            -o "${partial}" "${url}"
    else
        wget --continue -O "${partial}" "${url}"
    fi
    mv -f -- "${partial}" "${output}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'Dry run complete; no files were downloaded.\n'
else
    printf 'Model weights are available in %s\n' "${OUTPUT_DIR}"
fi
