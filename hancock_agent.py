#!/usr/bin/env python3
"""
Hancock Agent — CyberViser AI-powered cybersecurity agent enhanced with
Grok-1 314B Mixture-of-Experts inference.

Two modes of operation:

  python hancock_agent.py                → interactive CLI chat
  HANCOCK_API_KEY=... python hancock_agent.py --server
                                       → REST API server (Flask, default port 5000)

Backends:

  HANCOCK_LLM_BACKEND=grok     → Grok-1 native (314B MoE, requires GPU)
  HANCOCK_LLM_BACKEND=ollama   → local Ollama server (default)
  HANCOCK_LLM_BACKEND=nvidia   → NVIDIA NIM cloud API
  HANCOCK_LLM_BACKEND=openai   → OpenAI API

CLI commands:

  /mode pentest | soc | auto | code | ciso | sigma | yara | ioc
  /clear          — clear conversation history
  /history        — show history
  /model <id>     — switch model
  /exit           — quit

Set your key (for cloud backends):

  export NVIDIA_API_KEY="nvapi-..."
  export OPENAI_API_KEY="sk-..."
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import textwrap
from typing import Optional

from hancock_constants import (
    AGENT_NAME,
    ALL_MODES,
    BACKEND_GROK,
    BACKEND_NVIDIA,
    BACKEND_OLLAMA,
    BACKEND_OPENAI,
    COMPANY,
    DEFAULT_MODE,
    MODE_AUTO,
    MODE_CISO,
    MODE_CODE,
    MODE_IOC,
    MODE_PENTEST,
    MODE_SIGMA,
    MODE_SOC,
    MODE_YARA,
    VERSION,
    require_openai,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# System prompts — one per specialist mode
# ---------------------------------------------------------------------------

PENTEST_SYSTEM = textwrap.dedent("""\
    You are Hancock, an elite penetration tester and offensive security specialist built by CyberViser, \
    enhanced with Grok-1 314B MoE reasoning.

    Your expertise covers:
    - Reconnaissance: OSINT, subdomain enumeration, port scanning (nmap, amass, subfinder)
    - Web Application Testing: SQLi, XSS, SSRF, auth bypass, IDOR, JWT attacks
    - Network Exploitation: Metasploit, lateral movement, credential attacks
    - Post-Exploitation: Privilege escalation, persistence, pivoting
    - Vulnerability Analysis: CVE research, CVSS scoring, PoC analysis
    - Reporting: PTES methodology, professional finding write-ups

    You operate STRICTLY within authorized scope. You always:
    1. Confirm authorization before suggesting active techniques
    2. Recommend responsible disclosure and remediation
    3. Reference real tools, commands, and CVEs with accuracy
    4. Provide actionable, technically precise answers

    You are Hancock. You are methodical, precise, and professional.""")

SOC_SYSTEM = textwrap.dedent("""\
    You are Hancock, an expert SOC Tier-2/3 analyst and incident responder built by CyberViser, \
    enhanced with Grok-1 314B MoE reasoning.

    Your expertise covers:
    - Alert Triage: Classify and prioritize SIEM/EDR/IDS alerts using MITRE ATT&CK mapping
    - Log Analysis: Windows Event Logs, Syslog, Apache/Nginx, firewall, DNS
    - SIEM Queries: Splunk SPL, Elastic KQL, Microsoft Sentinel KQL
    - Incident Response: NIST SP 800-61 / PICERL framework
    - Threat Hunting: Hypothesis-driven hunting, IOC sweeps, behavioral analytics
    - Detection Engineering: Sigma rules, YARA rules, custom alerts

    You always:
    1. Follow the PICERL framework for incident response
    2. Document findings with timestamps, evidence, and chain of custody
    3. Write precise detection logic with comments
    4. Escalate appropriately and communicate clearly

    You are Hancock. You are methodical, calm, and thorough.""")

AUTO_SYSTEM = textwrap.dedent("""\
    You are Hancock, an elite cybersecurity specialist built by CyberViser, \
    enhanced with Grok-1 314B MoE reasoning. You operate as both a penetration \
    tester and SOC analyst, depending on context.

    **Pentest Mode:** Reconnaissance, exploitation, post-exploitation, CVE analysis.
    **SOC Mode:** Alert triage, SIEM queries, incident response (PICERL), threat hunting.

    You always:
    - Operate within authorized scope
    - Follow PICERL for IR and PTES for pentesting
    - Provide accurate, actionable technical guidance
    - Reference real tools, real CVEs, and real detection logic

    You are Hancock. Built by CyberViser.""")

CODE_SYSTEM = textwrap.dedent("""\
    You are Hancock Code, CyberViser's expert security code assistant, \
    enhanced with Grok-1 314B MoE reasoning.

    Your specialties:
    - Security automation: scanners, parsers, log analyzers, alert enrichers
    - SIEM query writing: Splunk SPL, Elastic KQL, Sentinel KQL, Sigma YAML
    - Detection scripts: YARA rules, Suricata/Snort rules, custom IDS signatures
    - Secure code review: identify vulns (OWASP Top 10, CWE), suggest fixes
    - CTF solvers: rev, pwn, web, crypto

    You always:
    1. Add authorization/legal warnings to offensive tooling
    2. Include error handling, type hints, and docstrings
    3. Explain what the code does and any security implications

    You are Hancock Code. Precision over verbosity. Ship working code.""")

CISO_SYSTEM = textwrap.dedent("""\
    You are Hancock CISO, CyberViser's AI-powered CISO advisor, \
    enhanced with Grok-1 314B MoE reasoning.

    Your expertise covers:
    - Risk Management: NIST RMF, ISO 27001/27005, FAIR quantitative risk analysis
    - Compliance: SOC 2, ISO 27001, PCI-DSS, HIPAA, GDPR, NIST CSF 2.0, CIS Controls v8
    - Board Reporting: security posture summaries, KRI/KPI dashboards
    - Security Architecture: zero trust, CSPM, identity governance

    You always:
    1. Translate technical risk into business impact
    2. Prioritize by likelihood x impact x cost-to-remediate
    3. Align to the organization's risk appetite
    4. Provide executive-ready language

    You are Hancock CISO. You speak business and security fluently.""")

SIGMA_SYSTEM = textwrap.dedent("""\
    You are Hancock Sigma, CyberViser's detection engineer specializing in Sigma rules, \
    enhanced with Grok-1 314B MoE reasoning.

    You always:
    1. Output valid, well-formed SIGMA YAML with all required fields
    2. Include falsepositives and level
    3. Tag correctly with MITRE ATT&CK technique IDs
    4. Add a filter condition when detection is prone to noise
    5. Explain what it detects and tuning notes

    You are Hancock Sigma. Every rule you write is ready to deploy.""")

YARA_SYSTEM = textwrap.dedent("""\
    You are Hancock YARA, CyberViser's malware analyst and detection engineer, \
    enhanced with Grok-1 314B MoE reasoning.

    You always:
    1. Output a complete, syntactically valid YARA rule with meta
    2. Use multiple string conditions to reduce false positives
    3. Limit to relevant file type/size when possible
    4. Explain what it detects and list false positive sources

    You are Hancock YARA. Every rule is ready to run.""")

IOC_SYSTEM = textwrap.dedent("""\
    You are Hancock IOC, CyberViser's threat intelligence analyst, \
    enhanced with Grok-1 314B MoE reasoning.

    When given an indicator of compromise (IP, domain, URL, hash, or email), provide:
    - Indicator type and classification
    - Threat intelligence context (malware families, threat actors, campaigns)
    - MITRE ATT&CK techniques associated
    - Risk score (1-10) with justification
    - Recommended defensive actions
    - Relevant CVEs or GHSA advisories if applicable

    Format as a clear, structured threat intel report.""")

SYSTEMS: dict[str, str] = {
    MODE_PENTEST: PENTEST_SYSTEM,
    MODE_SOC: SOC_SYSTEM,
    MODE_AUTO: AUTO_SYSTEM,
    MODE_CODE: CODE_SYSTEM,
    MODE_CISO: CISO_SYSTEM,
    MODE_SIGMA: SIGMA_SYSTEM,
    MODE_YARA: YARA_SYSTEM,
    MODE_IOC: IOC_SYSTEM,
}

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
CODER_MODEL = os.getenv("OLLAMA_CODER_MODEL", "qwen2.5-coder:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MODELS = {
    "llama3.1": "llama3.1:8b",
    "llama3.2": "llama3.2:3b",
    "mistral": "mistral:7b",
    "qwen-coder": "qwen2.5-coder:7b",
    "gemma3": "gemma3:12b",
    "nim-mistral": "mistralai/mistral-7b-instruct-v0.3",
    "nim-qwen": "qwen/qwen2.5-coder-32b-instruct",
    "nim-llama": "meta/llama-3.1-8b-instruct",
}

_banner_title = f"{COMPANY} — {AGENT_NAME} + Grok-1 314B MoE"
_banner_modes = "Pentest · SOC · CISO · Code · Sigma · YARA · IOC"
BANNER = f"""
╔══════════════════════════════════════════════════════════╗
║  ██╗  ██╗ █████╗ ███╗   ██╗ ██████╗ ██████╗  ██████╗██╗ ║
║  ██║  ██║██╔══██╗████╗  ██║██╔════╝██╔═══██╗██╔════╝██║ ║
║  ███████║███████║██╔██╗ ██║██║     ██║   ██║██║     ██║ ║
║  ██╔══██║██╔══██║██║╚██╗██║██║     ██║   ██║██║     ██╚╗║
║  ██║  ██║██║  ██║██║ ╚████║╚██████╗╚██████╔╝╚██████╗╚═╝║║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═════╝  ╚═════╝   ║
║  {_banner_title:<55s}║
║  {_banner_modes:<55s}║
╚══════════════════════════════════════════════════════════╝
  Modes : /mode {' | '.join(ALL_MODES)}
  Models: /model llama3.1 | mistral | qwen-coder | gemma3
  Other : /clear  /history  /exit
"""

# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def make_ollama_client() -> "OpenAI":
    require_openai(OpenAI)
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def make_nvidia_client(api_key: str) -> "OpenAI":
    require_openai(OpenAI)
    return OpenAI(base_url=NIM_BASE_URL, api_key=api_key)


def make_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("sk-your"):
        return None
    return OpenAI(api_key=key, organization=os.getenv("OPENAI_ORG_ID") or None)


# ---------------------------------------------------------------------------
# Grok-1 native backend wrapper
# ---------------------------------------------------------------------------


class GrokBackend:
    """Wraps :class:`hancock_runner.HancockGrokRunner` behind the same
    ``chat()`` interface used by OpenAI-compatible backends."""

    def __init__(self, runner):
        self._runner = runner

    def generate(self, messages: list[dict], max_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[System]\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"[User]\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"[{AGENT_NAME}]\n{content}\n")
        prompt_parts.append(f"[{AGENT_NAME}]\n")
        prompt = "\n".join(prompt_parts)
        return self._runner.generate(
            prompt,
            max_len=max_tokens,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------


def chat_openai(client: "OpenAI", history: list[dict], model: str,
                system_prompt: str, stream: bool = True) -> str:
    messages = [{"role": "system", "content": system_prompt}] + history
    if stream:
        response_text = ""
        print(f"\n\033[1;32m{AGENT_NAME}:\033[0m ", end="", flush=True)
        stream_resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=1024,
            temperature=0.7, top_p=0.95, stream=True,
        )
        for chunk in stream_resp:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                print(delta, end="", flush=True)
                response_text += delta
        print()
        return response_text
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=1024,
        temperature=0.7, top_p=0.95,
    )
    return resp.choices[0].message.content


def chat_grok(backend: GrokBackend, history: list[dict],
              system_prompt: str) -> str:
    messages = [{"role": "system", "content": system_prompt}] + history
    print(f"\n\033[1;32m{AGENT_NAME}:\033[0m ", end="", flush=True)
    response = backend.generate(messages)
    print(response)
    return response


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------


def run_cli(client, model: str, *, backend_name: str, grok_backend: Optional[GrokBackend] = None):
    print(BANNER)
    print(f"  Backend: {backend_name}")
    print(f"  Model  : {model}")
    print(f"  Mode   : {DEFAULT_MODE}")
    print()

    history: list[dict] = []
    current_mode = DEFAULT_MODE

    while True:
        try:
            user_input = input("\033[1;34m[You]\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[{AGENT_NAME}] Signing off. Stay in scope.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"[{AGENT_NAME}] Signing off. Stay in scope.")
            break

        if user_input == "/clear":
            history.clear()
            print(f"[{AGENT_NAME}] Conversation cleared.")
            continue

        if user_input == "/history":
            for i, m in enumerate(history):
                role = m["role"].upper()
                print(f"  [{i}] {role}: {m['content'][:80]}...")
            continue

        if user_input.startswith("/mode"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in SYSTEMS:
                current_mode = parts[1]
                history.clear()
                labels = {
                    MODE_PENTEST: "Pentest Specialist 🔴",
                    MODE_SOC: "SOC Analyst 🔵",
                    MODE_AUTO: "Auto (Pentest+SOC) ⚡",
                    MODE_CODE: "Code Assistant 💻",
                    MODE_CISO: "CISO Advisor 👔",
                    MODE_SIGMA: "Sigma Rules 🔍",
                    MODE_YARA: "YARA Rules 🦠",
                    MODE_IOC: "IOC Enrichment 🔎",
                }
                print(f"[{AGENT_NAME}] Switched to {labels.get(current_mode, current_mode)} — history cleared.")
            else:
                print(f"[{AGENT_NAME}] Usage: /mode {' | '.join(ALL_MODES)}")
            continue

        if user_input.startswith("/model "):
            alias = user_input[7:].strip()
            model = MODELS.get(alias, alias)
            print(f"[{AGENT_NAME}] Switched to model: {model}")
            continue

        history.append({"role": "user", "content": user_input})

        try:
            if grok_backend is not None:
                response = chat_grok(grok_backend, history, SYSTEMS[current_mode])
            else:
                response = chat_openai(client, history, model, SYSTEMS[current_mode])
            history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\033[1;31m[Error]\033[0m {e}")
            history.pop()


# ---------------------------------------------------------------------------
# REST API server
# ---------------------------------------------------------------------------


def build_app(client, model: str, *, grok_backend: Optional[GrokBackend] = None):
    """Build and return the Flask app."""
    try:
        from flask import Flask, Response, jsonify, request, stream_with_context
    except ImportError as exc:
        raise ImportError("Flask is required for server mode. Run: pip install flask") from exc

    app = Flask("hancock")

    import threading
    _metrics_lock = threading.Lock()
    _metrics: dict = {
        "requests_total": 0,
        "errors_total": 0,
        "requests_by_endpoint": {},
        "requests_by_mode": {},
    }

    def _inc(key: str, label: str = ""):
        with _metrics_lock:
            if label:
                _metrics[key][label] = _metrics[key].get(label, 0) + 1
            else:
                _metrics[key] += 1

    _HANCOCK_API_KEY = os.getenv("HANCOCK_API_KEY", "")
    _rate_counts: dict = {}
    _RATE_LIMIT = int(os.getenv("HANCOCK_RATE_LIMIT", "60"))
    _RATE_WINDOW = 60

    def _check_auth_and_rate():
        import time as _time
        now = _time.time()
        ip = request.remote_addr or "unknown"
        timestamps = [t for t in _rate_counts.get(ip, []) if now - t < _RATE_WINDOW]
        if len(timestamps) >= _RATE_LIMIT:
            return False, "Rate limit exceeded", 0
        timestamps.append(now)
        _rate_counts[ip] = timestamps
        if len(_rate_counts) > 10_000:
            cutoff = now - _RATE_WINDOW
            for k, v in list(_rate_counts.items()):
                if not v or v[-1] < cutoff:
                    del _rate_counts[k]
        remaining = max(0, _RATE_LIMIT - len(timestamps))
        if _HANCOCK_API_KEY:
            auth = request.headers.get("Authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(token, _HANCOCK_API_KEY):
                return False, "Unauthorized", remaining
        return True, "", remaining

    def _api_generate(messages: list[dict], max_tokens: int = 1024,
                      temperature: float = 0.7) -> str:
        if grok_backend is not None:
            return grok_backend.generate(messages, max_tokens=max_tokens,
                                         temperature=temperature)
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, top_p=0.95,
        )
        return resp.choices[0].message.content

    @app.after_request
    def _add_rate_headers(response):
        import time as _time
        ip = request.remote_addr or "unknown"
        now = _time.time()
        recent = [t for t in _rate_counts.get(ip, []) if now - t < _RATE_WINDOW]
        remaining = max(0, _RATE_LIMIT - len(recent))
        response.headers["X-RateLimit-Limit"] = str(_RATE_LIMIT)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = "60s"
        return response

    # -- Health ---------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "agent": AGENT_NAME,
            "version": VERSION,
            "model": model,
            "company": COMPANY,
            "grok_backend": grok_backend is not None,
            "modes": ALL_MODES,
            "endpoints": [
                "/v1/chat", "/v1/ask", "/v1/triage", "/v1/hunt",
                "/v1/respond", "/v1/code", "/v1/ciso", "/v1/sigma",
                "/v1/yara", "/v1/ioc", "/v1/agents", "/v1/webhook",
                "/metrics",
            ],
        })

    # -- Metrics --------------------------------------------------------------

    @app.route("/metrics", methods=["GET"])
    def metrics_endpoint():
        with _metrics_lock:
            snap = dict(_metrics)
        lines = [
            "# HELP hancock_requests_total Total API requests",
            "# TYPE hancock_requests_total counter",
            f'hancock_requests_total {snap["requests_total"]}',
            "# HELP hancock_errors_total Total errors",
            "# TYPE hancock_errors_total counter",
            f'hancock_errors_total {snap["errors_total"]}',
        ]
        for ep, cnt in snap.get("requests_by_endpoint", {}).items():
            lines.append(f'hancock_requests_by_endpoint{{endpoint="{ep}"}} {cnt}')
        for m, cnt in snap.get("requests_by_mode", {}).items():
            lines.append(f'hancock_requests_by_mode{{mode="{m}"}} {cnt}')
        return Response("\n".join(lines) + "\n", mimetype="text/plain; version=0.0.4")

    # -- Agents ---------------------------------------------------------------

    @app.route("/v1/agents", methods=["GET"])
    def agents_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/agents")
        return jsonify({
            "agents": {name: prompt for name, prompt in SYSTEMS.items()},
            "default_mode": DEFAULT_MODE,
            "model": model,
        })

    # -- Chat -----------------------------------------------------------------

    @app.route("/v1/chat", methods=["POST"])
    def chat_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/chat")
        data = request.get_json(force=True)
        user_msg = data.get("message", "")
        history = data.get("history", [])
        stream = data.get("stream", False)
        mode = data.get("mode", DEFAULT_MODE)
        if not user_msg:
            _inc("errors_total")
            return jsonify({"error": "message required"}), 400
        if mode not in SYSTEMS:
            _inc("errors_total")
            return jsonify({"error": f"invalid mode '{mode}'"}), 400
        if not isinstance(history, list):
            _inc("errors_total")
            return jsonify({"error": "history must be a list"}), 400
        _inc("requests_by_mode", mode)
        system = SYSTEMS[mode]
        history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": system}] + history

        if stream and grok_backend is None:
            def generate():
                full = ""
                stream_resp = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=1024,
                    temperature=0.7, top_p=0.95, stream=True,
                )
                for chunk in stream_resp:
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        full += delta
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                yield f"data: {json.dumps({'done': True, 'response': full})}\n\n"
            return Response(stream_with_context(generate()), mimetype="text/event-stream")

        response_text = _api_generate(messages)
        if not response_text:
            return jsonify({"error": "model returned empty response"}), 502
        return jsonify({"response": response_text, "model": model, "mode": mode})

    # -- Ask ------------------------------------------------------------------

    @app.route("/v1/ask", methods=["POST"])
    def ask_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/ask")
        data = request.get_json(force=True)
        question = data.get("question", "")
        mode = data.get("mode", DEFAULT_MODE)
        if not question:
            _inc("errors_total")
            return jsonify({"error": "question required"}), 400
        if mode not in SYSTEMS:
            _inc("errors_total")
            return jsonify({"error": f"invalid mode '{mode}'"}), 400
        messages = [
            {"role": "system", "content": SYSTEMS[mode]},
            {"role": "user", "content": question},
        ]
        answer = _api_generate(messages)
        return jsonify({"answer": answer, "model": model, "mode": mode})

    # -- Triage ---------------------------------------------------------------

    @app.route("/v1/triage", methods=["POST"])
    def triage_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/triage")
        _inc("requests_by_mode", MODE_SOC)
        data = request.get_json(force=True)
        alert = data.get("alert", "")
        if not alert:
            _inc("errors_total")
            return jsonify({"error": "alert required"}), 400
        prompt = (
            f"Triage the following security alert. Classify severity, "
            f"identify MITRE ATT&CK technique(s), determine TP/FP likelihood, "
            f"list containment actions, and recommend next steps.\n\nAlert:\n{alert}"
        )
        messages = [
            {"role": "system", "content": SOC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=1200, temperature=0.4)
        return jsonify({"triage": result, "model": model})

    # -- Hunt -----------------------------------------------------------------

    @app.route("/v1/hunt", methods=["POST"])
    def hunt_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/hunt")
        _inc("requests_by_mode", MODE_SOC)
        data = request.get_json(force=True)
        target = data.get("target", "")
        siem = data.get("siem", "splunk")
        if not target:
            _inc("errors_total")
            return jsonify({"error": "target required"}), 400
        prompt = (
            f"Generate a {siem.upper()} threat hunting query for: {target}\n"
            f"Include the query, data sources, expected fields, and MITRE ATT&CK mapping."
        )
        messages = [
            {"role": "system", "content": SOC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=1200, temperature=0.4)
        return jsonify({"query": result, "siem": siem, "model": model})

    # -- Respond (IR) ---------------------------------------------------------

    @app.route("/v1/respond", methods=["POST"])
    def respond_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/respond")
        _inc("requests_by_mode", MODE_SOC)
        data = request.get_json(force=True)
        incident_type = data.get("incident", "")
        if not incident_type:
            _inc("errors_total")
            return jsonify({"error": "incident required"}), 400
        prompt = (
            f"Provide a PICERL incident response playbook for: {incident_type}\n"
            f"Cover each phase with specific actions, tools, evidence, and communication steps."
        )
        messages = [
            {"role": "system", "content": SOC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=1500, temperature=0.4)
        return jsonify({"playbook": result, "incident": incident_type, "model": model})

    # -- Code -----------------------------------------------------------------

    @app.route("/v1/code", methods=["POST"])
    def code_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/code")
        _inc("requests_by_mode", MODE_CODE)
        data = request.get_json(force=True)
        task = data.get("task", "")
        language = data.get("language", "")
        if not task:
            _inc("errors_total")
            return jsonify({"error": "task required"}), 400
        lang_hint = f" Write in {language}." if language else ""
        prompt = f"{task}{lang_hint}\nProvide working, production-ready code."
        messages = [
            {"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=2048, temperature=0.2)
        return jsonify({"code": result, "model": model, "language": language or "auto"})

    # -- CISO -----------------------------------------------------------------

    @app.route("/v1/ciso", methods=["POST"])
    def ciso_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/ciso")
        _inc("requests_by_mode", MODE_CISO)
        data = request.get_json(force=True)
        question = data.get("question", "") or data.get("query", "")
        context = data.get("context", "")
        output_fmt = data.get("output", "advice")
        if not question:
            _inc("errors_total")
            return jsonify({"error": "question required"}), 400
        hints = {
            "report": "Format as a structured risk report.",
            "gap-analysis": "Format as a gap analysis table.",
            "board-summary": "Format as a board-ready executive summary (max 300 words).",
            "advice": "",
        }
        hint = hints.get(output_fmt, "")
        ctx_line = f"\nOrganisation context: {context}" if context else ""
        prompt = f"{question}{ctx_line}\n{hint}".strip()
        messages = [
            {"role": "system", "content": CISO_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=2048, temperature=0.3)
        if not result:
            _inc("errors_total")
            return jsonify({"error": "model returned empty response"}), 502
        return jsonify({"advice": result, "output": output_fmt, "model": model})

    # -- Sigma ----------------------------------------------------------------

    @app.route("/v1/sigma", methods=["POST"])
    def sigma_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/sigma")
        _inc("requests_by_mode", MODE_SIGMA)
        data = request.get_json(force=True)
        description = data.get("description", "") or data.get("ttp", "")
        logsource = data.get("logsource", "")
        technique = data.get("technique", "")
        if not description:
            _inc("errors_total")
            return jsonify({"error": "description required"}), 400
        hints = []
        if logsource:
            hints.append(f"Target log source: {logsource}.")
        if technique:
            hints.append(f"MITRE ATT&CK technique: {technique}.")
        prompt = (
            f"Write a production-ready Sigma rule for:\n{description}\n"
            f"{' '.join(hints)}\nOutput YAML rule then explanation."
        ).strip()
        messages = [
            {"role": "system", "content": SIGMA_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=2048, temperature=0.2)
        if not result:
            _inc("errors_total")
            return jsonify({"error": "model returned empty response"}), 502
        return jsonify({"rule": result, "logsource": logsource or "auto",
                        "technique": technique or "auto", "model": model})

    # -- YARA -----------------------------------------------------------------

    @app.route("/v1/yara", methods=["POST"])
    def yara_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/yara")
        _inc("requests_by_mode", MODE_YARA)
        data = request.get_json(force=True)
        description = data.get("description", "") or data.get("malware", "")
        file_type = data.get("file_type", "")
        sample_hash = data.get("hash", "")
        if not description:
            _inc("errors_total")
            return jsonify({"error": "description required"}), 400
        hints = []
        if file_type:
            hints.append(f"Target file type: {file_type}.")
        if sample_hash:
            hints.append(f"Known sample hash: {sample_hash}.")
        prompt = (
            f"Write a production-ready YARA rule for:\n{description}\n"
            f"{' '.join(hints)}\nOutput rule then explanation."
        ).strip()
        messages = [
            {"role": "system", "content": YARA_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=2048, temperature=0.2)
        if not result:
            _inc("errors_total")
            return jsonify({"error": "model returned empty response"}), 502
        return jsonify({"rule": result, "file_type": file_type or "auto", "model": model})

    # -- IOC ------------------------------------------------------------------

    @app.route("/v1/ioc", methods=["POST"])
    def ioc_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/ioc")
        _inc("requests_by_mode", MODE_IOC)
        data = request.get_json(force=True)
        indicator = (data.get("indicator") or data.get("ioc") or "").strip()
        ioc_type = data.get("type", "auto")
        context = data.get("context", "")
        if not indicator:
            _inc("errors_total")
            return jsonify({"error": "indicator required"}), 400
        prompt = f"Indicator: {indicator}\nType: {ioc_type}\n"
        if context:
            prompt += f"Additional context: {context}\n"
        prompt += "\nProvide a full threat intelligence enrichment report."
        messages = [
            {"role": "system", "content": IOC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        result = _api_generate(messages, max_tokens=1000, temperature=0.3)
        return jsonify({"indicator": indicator, "type": ioc_type,
                        "report": result, "model": model})

    # -- Webhook --------------------------------------------------------------

    @app.route("/v1/webhook", methods=["POST"])
    def webhook_endpoint():
        ok, err, _ = _check_auth_and_rate()
        if not ok:
            _inc("errors_total")
            return jsonify({"error": err}), 401 if "Unauthorized" in err else 429
        _inc("requests_total")
        _inc("requests_by_endpoint", "/v1/webhook")
        _inc("requests_by_mode", MODE_SOC)

        _WEBHOOK_SECRET = os.getenv("HANCOCK_WEBHOOK_SECRET", "")
        if _WEBHOOK_SECRET:
            import hashlib as _hashlib
            sig_header = request.headers.get("X-Hancock-Signature", "")
            body_bytes = request.get_data()
            expected = "sha256=" + hmac.new(
                _WEBHOOK_SECRET.encode(), body_bytes, _hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(sig_header, expected):
                _inc("errors_total")
                return jsonify({"error": "Invalid webhook signature"}), 401

        data = request.get_json(force=True)
        alert = data.get("alert", "")
        source = data.get("source", "unknown")
        severity = data.get("severity", "unknown")
        if not alert:
            _inc("errors_total")
            return jsonify({"error": "alert required"}), 400
        prompt = (
            f"[WEBHOOK ALERT from {source.upper()} | Severity: {severity.upper()}]\n"
            f"Triage this alert. Classify severity, map to MITRE ATT&CK, determine "
            f"TP/FP, list containment actions.\n\nAlert:\n{alert}"
        )
        messages = [
            {"role": "system", "content": SOC_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        triage_text = _api_generate(messages, max_tokens=1200, temperature=0.4)
        return jsonify({
            "status": "triaged",
            "source": source,
            "severity": severity,
            "triage": triage_text,
            "model": model,
        })

    return app


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def run_server(client, model: str, port: int, *,
               grok_backend: Optional[GrokBackend] = None):
    if not os.getenv("HANCOCK_API_KEY", "").strip():
        raise RuntimeError(
            "HANCOCK_API_KEY must be set before starting Hancock server mode."
        )
    app = build_app(client, model, grok_backend=grok_backend)
    print(f"\n[{COMPANY}] {AGENT_NAME} API server starting on port {port}")
    print(f"  POST /v1/chat     — conversational (mode: {' | '.join(ALL_MODES)})")
    print(f"  POST /v1/ask      — single question")
    print(f"  POST /v1/triage   — SOC alert triage")
    print(f"  POST /v1/hunt     — threat hunting query generator")
    print(f"  POST /v1/respond  — IR playbook (PICERL)")
    print(f"  POST /v1/code     — security code generation")
    print(f"  POST /v1/ciso     — CISO advisory")
    print(f"  POST /v1/sigma    — Sigma rule generator")
    print(f"  POST /v1/yara     — YARA rule generator")
    print(f"  POST /v1/ioc      — IOC enrichment")
    print(f"  POST /v1/webhook  — SIEM push webhook")
    print(f"  GET  /health      — status check")
    print(f"  GET  /metrics     — Prometheus metrics\n")
    app.run(host="0.0.0.0", port=port, debug=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=f"{AGENT_NAME} — {COMPANY} AI Cybersecurity Agent + Grok-1",
    )
    parser.add_argument("--server", action="store_true", help="Run as REST API server")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--api-key", type=str, default=os.getenv("NVIDIA_API_KEY", ""))
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--backend", type=str,
        default=os.getenv("HANCOCK_LLM_BACKEND", BACKEND_OLLAMA),
        choices=[BACKEND_GROK, BACKEND_OLLAMA, BACKEND_NVIDIA, BACKEND_OPENAI],
        help="LLM backend to use",
    )
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/",
                        help="Grok-1 checkpoint path (only for grok backend)")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer.model",
                        help="Tokenizer path (only for grok backend)")
    args = parser.parse_args()

    backend = args.backend.lower()
    client = None
    model = ""
    grok_backend = None

    if backend == BACKEND_GROK:
        from hancock_runner import HancockGrokRunner
        runner = HancockGrokRunner(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
        )
        runner.initialize()
        grok_backend = GrokBackend(runner)
        model = "grok-1-314b"
        print(f"[{AGENT_NAME}] Using Grok-1 native backend (314B MoE).")
    elif backend == BACKEND_OLLAMA:
        client = make_ollama_client()
        model = args.model or DEFAULT_MODEL
        print(f"[{AGENT_NAME}] Using Ollama backend ({OLLAMA_BASE_URL}).")
    elif backend == BACKEND_NVIDIA and args.api_key:
        client = make_nvidia_client(args.api_key)
        model = args.model or "mistralai/mistral-7b-instruct-v0.3"
        print(f"[{AGENT_NAME}] Using NVIDIA NIM backend.")
    elif backend == BACKEND_OPENAI:
        client = make_openai_client()
        if not client:
            sys.exit(
                "ERROR: OpenAI backend requires OPENAI_API_KEY env var.\n"
                "  export OPENAI_API_KEY='sk-...'"
            )
        model = args.model or OPENAI_MODEL
        print(f"[{AGENT_NAME}] Using OpenAI backend.")
    else:
        sys.exit(
            "ERROR: No backend configured.\n"
            f"  Set HANCOCK_LLM_BACKEND to one of: {', '.join([BACKEND_GROK, BACKEND_OLLAMA, BACKEND_NVIDIA, BACKEND_OPENAI])}\n"
            "  For Grok-1: --backend grok\n"
            "  For Ollama: install Ollama and set HANCOCK_LLM_BACKEND=ollama\n"
            "  For cloud:  set OPENAI_API_KEY or NVIDIA_API_KEY"
        )

    if args.server:
        run_server(client, model, args.port, grok_backend=grok_backend)
    else:
        run_cli(client, model, backend_name=backend, grok_backend=grok_backend)


if __name__ == "__main__":
    main()
