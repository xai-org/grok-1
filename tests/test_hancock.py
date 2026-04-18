"""Unit tests for the Hancock cybersecurity agent modules."""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# hancock_constants tests
# ---------------------------------------------------------------------------


class TestHancockConstants(unittest.TestCase):
    """Tests for hancock_constants module."""

    def test_all_modes_list(self):
        from hancock_constants import ALL_MODES, DEFAULT_MODE
        self.assertIn("pentest", ALL_MODES)
        self.assertIn("soc", ALL_MODES)
        self.assertIn("auto", ALL_MODES)
        self.assertIn("code", ALL_MODES)
        self.assertIn("ciso", ALL_MODES)
        self.assertIn("sigma", ALL_MODES)
        self.assertIn("yara", ALL_MODES)
        self.assertIn("ioc", ALL_MODES)
        self.assertEqual(len(ALL_MODES), 8)
        self.assertEqual(DEFAULT_MODE, "auto")

    def test_version_string(self):
        from hancock_constants import VERSION
        parts = VERSION.split(".")
        self.assertEqual(len(parts), 3)

    def test_require_openai_raises(self):
        from hancock_constants import require_openai, OPENAI_IMPORT_ERROR_MSG
        with self.assertRaises(ImportError) as ctx:
            require_openai(None)
        self.assertIn("openai", str(ctx.exception).lower())

    def test_require_openai_passes(self):
        from hancock_constants import require_openai
        require_openai(object)  # any non-None should pass


# ---------------------------------------------------------------------------
# hancock_agent tests
# ---------------------------------------------------------------------------


class TestHancockAgentSystemPrompts(unittest.TestCase):
    """Tests for system prompts and mode configuration."""

    def test_systems_dict_has_all_modes(self):
        from hancock_agent import SYSTEMS
        from hancock_constants import ALL_MODES
        for mode in ALL_MODES:
            self.assertIn(mode, SYSTEMS)
            self.assertIsInstance(SYSTEMS[mode], str)
            self.assertGreater(len(SYSTEMS[mode]), 50)

    def test_systems_mention_hancock(self):
        from hancock_agent import SYSTEMS
        for mode, prompt in SYSTEMS.items():
            self.assertIn("Hancock", prompt, f"Mode {mode!r} missing Hancock identity")

    def test_systems_mention_grok(self):
        from hancock_agent import SYSTEMS
        for mode, prompt in SYSTEMS.items():
            self.assertIn("Grok-1", prompt, f"Mode {mode!r} missing Grok-1 reference")


class TestGrokBackend(unittest.TestCase):
    """Tests for the GrokBackend adapter class."""

    def test_generate_calls_runner(self):
        from hancock_agent import GrokBackend
        mock_runner = MagicMock()
        mock_runner.generate.return_value = "test response"
        backend = GrokBackend(mock_runner)

        messages = [
            {"role": "system", "content": "You are Hancock."},
            {"role": "user", "content": "What is CVE-2024-3094?"},
        ]
        result = backend.generate(messages)
        self.assertEqual(result, "test response")
        mock_runner.generate.assert_called_once()
        call_args = mock_runner.generate.call_args
        prompt = call_args[1].get("prompt") or call_args[0][0]
        self.assertIn("CVE-2024-3094", prompt)


class TestFlaskApp(unittest.TestCase):
    """Tests for the Flask REST API endpoints."""

    def setUp(self):
        """Create a test Flask app with a mocked LLM client."""
        from hancock_agent import build_app

        # Mock OpenAI client
        self.mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mock Hancock response"
        self.mock_client.chat.completions.create.return_value = mock_response

        self.app = build_app(self.mock_client, "test-model")
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["agent"], "Hancock")
        self.assertIn("modes", data)
        self.assertIn("endpoints", data)

    def test_metrics_endpoint(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"hancock_requests_total", resp.data)

    def test_agents_endpoint(self):
        resp = self.client.get("/v1/agents")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("agents", data)
        self.assertIn("pentest", data["agents"])

    def test_ask_endpoint(self):
        resp = self.client.post("/v1/ask", json={"question": "What is XSS?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Mock Hancock response")

    def test_ask_missing_question(self):
        resp = self.client.post("/v1/ask", json={})
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_ask_invalid_mode(self):
        resp = self.client.post("/v1/ask", json={
            "question": "What is XSS?",
            "mode": "invalid_mode",
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_chat_endpoint(self):
        resp = self.client.post("/v1/chat", json={
            "message": "Tell me about nmap",
            "mode": "pentest",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("response", data)
        self.assertEqual(data["mode"], "pentest")

    def test_chat_invalid_mode(self):
        resp = self.client.post("/v1/chat", json={
            "message": "test",
            "mode": "invalid_mode",
        })
        self.assertEqual(resp.status_code, 400)

    def test_chat_missing_message(self):
        resp = self.client.post("/v1/chat", json={})
        self.assertEqual(resp.status_code, 400)

    def test_triage_endpoint(self):
        resp = self.client.post("/v1/triage", json={
            "alert": "Mimikatz detected on DC01",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("triage", data)

    def test_triage_missing_alert(self):
        resp = self.client.post("/v1/triage", json={})
        self.assertEqual(resp.status_code, 400)

    def test_hunt_endpoint(self):
        resp = self.client.post("/v1/hunt", json={
            "target": "lateral movement via PsExec",
            "siem": "splunk",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("query", data)
        self.assertEqual(data["siem"], "splunk")

    def test_hunt_missing_target(self):
        resp = self.client.post("/v1/hunt", json={})
        self.assertEqual(resp.status_code, 400)

    def test_respond_endpoint(self):
        resp = self.client.post("/v1/respond", json={"incident": "ransomware"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("playbook", data)

    def test_respond_missing_incident(self):
        resp = self.client.post("/v1/respond", json={})
        self.assertEqual(resp.status_code, 400)

    def test_code_endpoint(self):
        resp = self.client.post("/v1/code", json={
            "task": "Write a port scanner",
            "language": "python",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("code", data)

    def test_code_missing_task(self):
        resp = self.client.post("/v1/code", json={})
        self.assertEqual(resp.status_code, 400)

    def test_ciso_endpoint(self):
        resp = self.client.post("/v1/ciso", json={
            "question": "Top 5 risks for the board",
            "output": "board-summary",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("advice", data)

    def test_ciso_missing_question(self):
        resp = self.client.post("/v1/ciso", json={})
        self.assertEqual(resp.status_code, 400)

    def test_sigma_endpoint(self):
        resp = self.client.post("/v1/sigma", json={
            "description": "Detect LSASS memory dump",
            "logsource": "windows sysmon",
            "technique": "T1003.001",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("rule", data)

    def test_sigma_missing_description(self):
        resp = self.client.post("/v1/sigma", json={})
        self.assertEqual(resp.status_code, 400)

    def test_yara_endpoint(self):
        resp = self.client.post("/v1/yara", json={
            "description": "Cobalt Strike beacon",
            "file_type": "PE",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("rule", data)

    def test_yara_missing_description(self):
        resp = self.client.post("/v1/yara", json={})
        self.assertEqual(resp.status_code, 400)

    def test_ioc_endpoint(self):
        resp = self.client.post("/v1/ioc", json={
            "indicator": "185.220.101.35",
            "type": "ip",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("report", data)

    def test_ioc_missing_indicator(self):
        resp = self.client.post("/v1/ioc", json={})
        self.assertEqual(resp.status_code, 400)

    def test_webhook_endpoint(self):
        resp = self.client.post("/v1/webhook", json={
            "alert": "Suspicious login from TOR exit node",
            "source": "splunk",
            "severity": "high",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "triaged")

    def test_webhook_missing_alert(self):
        resp = self.client.post("/v1/webhook", json={})
        self.assertEqual(resp.status_code, 400)

    def test_rate_limit_headers(self):
        resp = self.client.get("/health")
        self.assertIn("X-RateLimit-Limit", resp.headers)
        self.assertIn("X-RateLimit-Remaining", resp.headers)

    def test_failed_auth_requests_are_rate_limited(self):
        from hancock_agent import build_app

        with patch.dict(
            os.environ,
            {"HANCOCK_API_KEY": "secret-token", "HANCOCK_RATE_LIMIT": "2"},
            clear=False,
        ):
            app = build_app(self.mock_client, "test-model")
            client = app.test_client()

            for _ in range(2):
                resp = client.post(
                    "/v1/ask",
                    json={"question": "What is XSS?"},
                    headers={"Authorization": "Bearer wrong-token"},
                )
                self.assertEqual(resp.status_code, 401)

            resp = client.post(
                "/v1/ask",
                json={"question": "What is XSS?"},
                headers={"Authorization": "Bearer wrong-token"},
            )
            self.assertEqual(resp.status_code, 429)

    def test_run_server_requires_api_key(self):
        from hancock_agent import run_server

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                RuntimeError,
                "HANCOCK_API_KEY must be set before starting Hancock server mode.",
            ):
                run_server(self.mock_client, "test-model", 5000)


class TestFlaskAppWithGrokBackend(unittest.TestCase):
    """Tests for Flask API endpoints using the Grok-1 backend."""

    def setUp(self):
        from hancock_agent import build_app, GrokBackend
        mock_runner = MagicMock()
        mock_runner.generate.return_value = "Grok-1 response"
        self.grok_backend = GrokBackend(mock_runner)
        self.app = build_app(None, "grok-1-314b", grok_backend=self.grok_backend)
        self.client = self.app.test_client()

    def test_health_shows_grok_backend(self):
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertTrue(data["grok_backend"])

    def test_ask_uses_grok_backend(self):
        resp = self.client.post("/v1/ask", json={"question": "Explain CVE-2024-3094"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["answer"], "Grok-1 response")


# ---------------------------------------------------------------------------
# hancock_runner tests
# ---------------------------------------------------------------------------


class TestHancockGrokRunner(unittest.TestCase):
    """Tests for the HancockGrokRunner bridge class."""

    def test_runner_not_ready_before_init(self):
        from hancock_runner import HancockGrokRunner
        runner = HancockGrokRunner()
        self.assertFalse(runner.is_ready)

    def test_generate_raises_without_init(self):
        from hancock_runner import HancockGrokRunner
        runner = HancockGrokRunner()
        with self.assertRaises(RuntimeError):
            runner.generate("test prompt")

    def test_defaults(self):
        from hancock_runner import HancockGrokRunner
        runner = HancockGrokRunner()
        self.assertEqual(runner.checkpoint_path, "./checkpoints/")
        self.assertEqual(runner.tokenizer_path, "./tokenizer.model")
        self.assertEqual(runner.max_len, 256)
        self.assertAlmostEqual(runner.temperature, 0.7)


if __name__ == "__main__":
    unittest.main()
