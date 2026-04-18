"""Shared constants for Hancock cybersecurity agent modules."""

VERSION = "1.0.0"
AGENT_NAME = "Hancock"
COMPANY = "CyberViser"

# Mode identifiers
MODE_PENTEST = "pentest"
MODE_SOC = "soc"
MODE_AUTO = "auto"
MODE_CODE = "code"
MODE_CISO = "ciso"
MODE_SIGMA = "sigma"
MODE_YARA = "yara"
MODE_IOC = "ioc"

ALL_MODES = [
    MODE_PENTEST,
    MODE_SOC,
    MODE_AUTO,
    MODE_CODE,
    MODE_CISO,
    MODE_SIGMA,
    MODE_YARA,
    MODE_IOC,
]

DEFAULT_MODE = MODE_AUTO

# Backend identifiers
BACKEND_GROK = "grok"
BACKEND_OLLAMA = "ollama"
BACKEND_NVIDIA = "nvidia"
BACKEND_OPENAI = "openai"

OPENAI_IMPORT_ERROR_MSG = "OpenAI client not installed. Run: pip install openai"


def require_openai(openai_cls):
    """Raise ImportError when the OpenAI dependency is missing."""
    if openai_cls is None:
        raise ImportError(OPENAI_IMPORT_ERROR_MSG)
