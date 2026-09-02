from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT_DIR / ".env"


def load_env(path: str | Path = DEFAULT_ENV_PATH) -> None:
    """Load KEY=VALUE pairs from a local .env file without overwriting env vars."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_secret(name: str, required: bool = False) -> str | None:
    load_env()
    value = os.environ.get(name)
    if required and not value:
        raise RuntimeError(
            f"{name} is required. Set it in your shell or in a local .env file."
        )
    return value


def get_openai_api_key(required: bool = False) -> str | None:
    return get_secret("OPENAI_API_KEY", required=required)


def get_hf_token(required: bool = False) -> str | None:
    return (
        get_secret("HF_TOKEN", required=False)
        or get_secret("HUGGINGFACE_TOKEN", required=required)
    )


def get_fmp_api_key(required: bool = False) -> str | None:
    return (
        get_secret("FMP_API_KEY", required=False)
        or get_secret("FINANCIAL_MODELING_PREP_API_KEY", required=required)
    )


def get_sec_user_agent(required: bool = False) -> str | None:
    """SEC EDGAR requires a descriptive User-Agent with contact info on every
    request (https://www.sec.gov/os/webmaster-faq#developers). Set
    SEC_USER_AGENT="YourApp yourname@example.com" in .env; sec13f/edgar.py
    falls back to a generic default if unset (works, but SEC recommends a
    real contact address).
    """
    return get_secret("SEC_USER_AGENT", required=required)


def get_openfigi_api_key(required: bool = False) -> str | None:
    """Optional. Without a key, OpenFIGI's /v3/mapping endpoint is still
    usable but rate-limited more strictly (see sec13f/mapping.py).
    """
    return get_secret("OPENFIGI_API_KEY", required=required)
