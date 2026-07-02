#!/usr/bin/env bash
# vvv THOG

_thog2_setup_main() {
    local recreate=false
    local venv_dir="${THOG2_VENV_DIR:-$HOME/.venvs/thog2}"
    local python_version="${THOG2_PYTHON_VERSION:-3.11}"
    local torch_index_url="${THOG2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    local uv_http_timeout="${THOG2_UV_HTTP_TIMEOUT:-600}"
    local uv_http_retries="${THOG2_UV_HTTP_RETRIES:-5}"
    local conda_deactivate_count=0

    while (($#)); do
        case "$1" in
            --recreate)
                recreate=true
                ;;
            -h|--help)
                cat <<EOF
Usage:
  source docs/THOG2_Stage_3_Setup_Environment.sh [--recreate]

Environment variables:
  THOG2_VENV_DIR         Dedicated venv path
                         Default: $HOME/.venvs/thog2
  THOG2_PYTHON_VERSION   Python version requested from uv
                         Default: 3.11
  THOG2_TORCH_INDEX_URL  PyTorch wheel index
                         Default: https://download.pytorch.org/whl/cu126
  THOG2_UV_HTTP_TIMEOUT  Per-request uv HTTP timeout in seconds
                         Default: 600
  THOG2_UV_HTTP_RETRIES  Number of uv HTTP retries
                         Default: 5
EOF
                return 0
                ;;
            *)
                printf 'ERROR: unknown argument: %s\n' "$1" >&2
                return 2
                ;;
        esac
        shift
    done

    if ! [[ "$uv_http_timeout" =~ ^[1-9][0-9]*$ ]]; then
        printf 'ERROR: THOG2_UV_HTTP_TIMEOUT must be a positive integer.\n' >&2
        return 2
    fi
    if ! [[ "$uv_http_retries" =~ ^[0-9]+$ ]]; then
        printf 'ERROR: THOG2_UV_HTTP_RETRIES must be a non-negative integer.\n' >&2
        return 2
    fi

    if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -F deactivate >/dev/null 2>&1; then
        printf 'Deactivating existing Python virtual environment: %s\n' "$VIRTUAL_ENV"
        deactivate
    fi

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        if [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
            if command -v conda >/dev/null 2>&1; then
                eval "$(command conda shell.bash hook)"
            fi
        fi
        if [[ "$(type -t conda 2>/dev/null || true)" != "function" ]]; then
            printf 'ERROR: Conda is active but its shell function is unavailable.\n' >&2
            return 1
        fi
        while [[ -n "${CONDA_PREFIX:-}" && "$conda_deactivate_count" -lt 10 ]]; do
            printf 'Deactivating Conda environment: %s\n' "$CONDA_PREFIX"
            conda deactivate || return 1
            conda_deactivate_count=$((conda_deactivate_count + 1))
        done
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            printf 'ERROR: Conda remains active after repeated deactivation.\n' >&2
            return 1
        fi
    fi

    if ! command -v uv >/dev/null 2>&1; then
        cat >&2 <<'EOF'
ERROR: uv is not installed or is unavailable after Conda deactivation.

Install uv outside Conda with:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then open a new shell and source this script again.
EOF
        return 1
    fi

    if [[ "$recreate" == true && -e "$venv_dir" ]]; then
        printf 'Removing existing THOG2 environment: %s\n' "$venv_dir"
        rm -rf -- "$venv_dir" || return 1
    fi

    if [[ -e "$venv_dir" && ! -x "$venv_dir/bin/python" ]]; then
        printf 'ERROR: %s exists but is not a valid virtual environment.\n' "$venv_dir" >&2
        printf 'Rerun with --recreate after checking the path.\n' >&2
        return 1
    fi

    if [[ ! -x "$venv_dir/bin/python" ]]; then
        mkdir -p -- "$(dirname "$venv_dir")" || return 1
        printf 'Creating dedicated THOG2 environment: %s\n' "$venv_dir"
        uv venv \
            --python "$python_version" \
            --prompt thog2 \
            "$venv_dir" || return 1
    elif ! grep -Fq 'VIRTUAL_ENV_PROMPT="thog2"' "$venv_dir/bin/activate"; then
        printf 'ERROR: the existing environment does not use the prompt thog2.\n' >&2
        printf 'Recreate it with:\n' >&2
        printf '  source docs/THOG2_Stage_3_Setup_Environment.sh --recreate\n' >&2
        return 1
    else
        printf 'Using existing THOG2 environment: %s\n' "$venv_dir"
    fi

    printf 'uv download timeout: %s seconds; retries: %s\n' \
        "$uv_http_timeout" \
        "$uv_http_retries"

    printf 'Installing NumPy from PyPI...\n'
    env \
        UV_HTTP_TIMEOUT="$uv_http_timeout" \
        UV_HTTP_RETRIES="$uv_http_retries" \
        uv pip install \
            --python "$venv_dir/bin/python" \
            --upgrade \
            numpy || return 1

    printf 'Installing PyTorch from: %s\n' "$torch_index_url"
    env \
        UV_HTTP_TIMEOUT="$uv_http_timeout" \
        UV_HTTP_RETRIES="$uv_http_retries" \
        uv pip install \
            --python "$venv_dir/bin/python" \
            --upgrade \
            --index-url "$torch_index_url" \
            torch || return 1

    unset VIRTUAL_ENV_DISABLE_PROMPT
    # shellcheck disable=SC1090
    source "$venv_dir/bin/activate" || return 1
    hash -r

    python - <<'PY'
import sys
import torch

print(f"Python:         {sys.version.split()[0]}")
print(f"Executable:     {sys.executable}")
print(f"PyTorch:        {torch.__version__}")
print(f"PyTorch CUDA:   {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: the thog2 environment is active, but PyTorch cannot access CUDA."
    )

print(f"GPU:            {torch.cuda.get_device_name(0)}")
x = torch.randn(512, 512, device="cuda")
y = x @ x
torch.cuda.synchronize()
print(f"CUDA smoke:     PASS ({tuple(y.shape)})")
PY
    local verification_status=$?

    if [[ "$verification_status" -ne 0 ]]; then
        printf '\nThe environment remains active for diagnosis.\n' >&2
        return "$verification_status"
    fi

    printf '\nTHOG2 environment ready. Current prompt should be (thog2).\n'
    printf 'Run the Stage 3 GPU gate with:\n'
    printf '  python tests/run_sheet_stage3_gpu_smoke.py \\\n'
    printf '    --evidence evidence/stage3_gpu_smoke.json\n'
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat >&2 <<'EOF'
ERROR: this script must be sourced so it can deactivate Conda and activate
       the THOG2 virtual environment in the current shell.

Run:
  source docs/THOG2_Stage_3_Setup_Environment.sh
EOF
    exit 2
fi

_thog2_setup_main "$@"
_thog2_setup_status=$?
unset -f _thog2_setup_main
return "$_thog2_setup_status"

# ^^^ THOG
