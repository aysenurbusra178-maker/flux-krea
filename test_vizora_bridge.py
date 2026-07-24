from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vizora_bridge import PromptReceipt, VizoraPromptClient


class PromptReceiptTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        receipt = PromptReceipt(
            generation_id="11111111-1111-4111-8111-111111111111",
            prompt_id="vizora-abc",
            prompt_hash="a" * 64,
            provider="flux-krea",
            media_type="image",
            positive_prompt="OBJECTIVE\n- portrait",
            negative_prompt="plastic skin",
            module_versions={"identity_skin_anatomy_master": "1.0.0"},
            validation_rules=["Reject identity drift"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            receipt.save(path)
            self.assertEqual(PromptReceipt.load(path), receipt)


class EndpointRestrictionTests(unittest.TestCase):
    def test_loopback_is_allowed(self) -> None:
        client = VizoraPromptClient(
            base_url="http://127.0.0.1:4317",
            token="x" * 24,
        )
        self.assertEqual(client.base_url, "http://127.0.0.1:4317")

    def test_private_lan_requires_opt_in(self) -> None:
        with self.assertRaises(ValueError):
            VizoraPromptClient(
                base_url="http://192.168.1.50:4317",
                token="x" * 24,
                allow_lan=False,
            )

        client = VizoraPromptClient(
            base_url="http://192.168.1.50:4317",
            token="x" * 24,
            allow_lan=True,
        )
        self.assertEqual(client.base_url, "http://192.168.1.50:4317")

    @patch.dict(os.environ, {}, clear=True)
    def test_short_token_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VizoraPromptClient(base_url="http://127.0.0.1:4317", token="short")


if __name__ == "__main__":
    unittest.main()
