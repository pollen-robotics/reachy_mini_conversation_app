#!/bin/bash
# reachy-stt-warm-load.sh
#
# ExecStartPost helper for the reachy-stt systemd unit.
#
# Reads /home/pollen/.robot_comic/.env to determine the active audio input
# backend. If it is one of the on-device STT backends (faster_whisper,
# local_stt, moonshine) it POSTs /preload to the reachy-stt service so that
# faster-whisper loads its model into memory before the first transcription
# request arrives.
#
# If the active backend is anything else (gemini_live_input, openai_realtime,
# etc.) the model is left unloaded (~30 MB idle resident) so we don't waste
# memory or load time when a remote speech path is in use.
#
# Why a separate script rather than an inline ExecStartPost= command?
# The decision logic (parse .env, compare, conditional curl) is easier to
# test, audit, and evolve as a standalone script than as a shell fragment
# embedded in the unit file.
#
# Exit behaviour:
#   Always exits 0. A curl failure is logged but must not prevent the service
#   from entering the "active" state — the app can still warm-load on demand
#   via the /preload endpoint.
#
# Usage: invoked automatically by systemd as ExecStartPost.
#        Can also be run manually for testing.

set -u
set -o pipefail
# NOTE: intentionally NOT set -e so we can swallow curl failures below.

# ENV_FILE and SOCKET honor pre-set values so the script is testable:
# tests prepend `ENV_FILE=...; SOCKET=...` before sourcing this file.
ENV_FILE="${ENV_FILE:-/home/pollen/.robot_comic/.env}"
SOCKET="${SOCKET:-/run/reachy-stt/reachy-stt.sock}"
PRELOAD_URL="http://localhost/preload"

# Backends that indicate on-device STT is active and the model should be
# pre-loaded. Case-insensitive comparison is applied below.
ON_DEVICE_BACKENDS="faster_whisper local_stt moonshine"

# ---------------------------------------------------------------------------
# Read the active input backend from .env
# ---------------------------------------------------------------------------
active_backend=""

if [[ -f "${ENV_FILE}" ]]; then
    # Strip comments, blank lines, and surrounding whitespace/quotes, then
    # extract REACHY_MINI_AUDIO_INPUT_BACKEND. Fall back to
    # REACHY_MINI_PIPELINE_MODE for legacy configs that predate the
    # composable pipeline refactor.
    while IFS= read -r line; do
        # Skip blank lines and comment lines
        [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue

        # Strip leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        if [[ "${line}" =~ ^REACHY_MINI_AUDIO_INPUT_BACKEND[[:space:]]*=[[:space:]]*(.*) ]]; then
            val="${BASH_REMATCH[1]}"
            # Strip surrounding quotes (single or double)
            val="${val#\'}" ; val="${val%\'}"
            val="${val#\"}" ; val="${val%\"}"
            active_backend="${val}"
            break
        fi
    done < "${ENV_FILE}"

    # Legacy fallback: REACHY_MINI_PIPELINE_MODE
    if [[ -z "${active_backend}" ]]; then
        while IFS= read -r line; do
            [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            if [[ "${line}" =~ ^REACHY_MINI_PIPELINE_MODE[[:space:]]*=[[:space:]]*(.*) ]]; then
                val="${BASH_REMATCH[1]}"
                val="${val#\'}" ; val="${val%\'}"
                val="${val#\"}" ; val="${val%\"}"
                active_backend="${val}"
                break
            fi
        done < "${ENV_FILE}"
    fi
else
    echo "reachy-stt-warm-load: ${ENV_FILE} not found — skipping warm-load" >&2
    exit 0
fi

if [[ -z "${active_backend}" ]]; then
    echo "reachy-stt-warm-load: no input backend configured in ${ENV_FILE} — skipping warm-load" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# Decide whether to warm-load
# ---------------------------------------------------------------------------
backend_lower="${active_backend,,}"  # bash 4+ lowercase
should_preload=0
for b in ${ON_DEVICE_BACKENDS}; do
    if [[ "${backend_lower}" == "${b}" ]]; then
        should_preload=1
        break
    fi
done

if [[ "${should_preload}" -eq 0 ]]; then
    echo "reachy-stt-warm-load: not warm-loading (active backend=${active_backend}, model stays idle)" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# POST /preload — swallow errors, always exit 0
# ---------------------------------------------------------------------------
echo "reachy-stt-warm-load: active backend=${active_backend} — sending /preload to ${SOCKET}" >&2

curl_output=$(
    curl \
        --unix-socket "${SOCKET}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{}' \
        "${PRELOAD_URL}" \
        --max-time 30 \
        --silent \
        --show-error \
        2>&1
)
# Capture curl's exit directly. `set -e` is not active, so a non-zero exit
# from the command substitution does NOT abort the script — we just record
# the status and decide what to log. An earlier `|| true` here masked $?
# so the failure branch never fired.
curl_exit=$?
if [[ "${curl_exit}" -ne 0 ]]; then
    echo "reachy-stt-warm-load: curl /preload failed (exit ${curl_exit}): ${curl_output}" >&2
    echo "reachy-stt-warm-load: model will load on first transcription request instead" >&2
else
    echo "reachy-stt-warm-load: /preload succeeded: ${curl_output}" >&2
fi

exit 0
