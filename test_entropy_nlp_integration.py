#!/usr/bin/env python3
import sys
import os
import json
import unittest
from pathlib import Path

# Add local_ai to path
sys.path.insert(0, str(Path(__file__).parent / "local_ai"))

class TestEntropyNLPIntegration(unittest.TestCase):
    def test_import(self):
        import entropy_nlp
        self.assertTrue(hasattr(entropy_nlp, "status"))

    def test_status(self):
        import entropy_nlp
        res = entropy_nlp.status()
        self.assertTrue(res["ok"])
        self.assertEqual(res["status"], "active")

    def test_diagnose_verbification(self):
        import entropy_nlp
        text = "A cat gentles an animal."
        diag = entropy_nlp.diagnose_text(text, profile="curriculum")
        self.assertFalse(diag.ok)
        self.assertTrue(any("gentles" in w for w in diag.warnings))

    def test_rerank(self):
        import entropy_nlp
        candidates = [
            "A cat is gentle.",
            "An animal is gentle.",
            "A cat gentles an animal."
        ]
        reranked = entropy_nlp.rerank_texts(
            candidates, 
            context="cat animal gentle", 
            profile="curriculum"
        )
        self.assertEqual(len(reranked), 3)
        # "gentles" should be last due to penalty
        self.assertEqual(reranked[-1]["text"], "A cat gentles an animal.")

    def test_adapter_integration(self):
        import local_ai_adapter
        adapter = local_ai_adapter.get_adapter()
        status = adapter.get_status()
        self.assertIn("entropy_nlp", status)
        self.assertTrue(status["entropy_nlp"]["ok"])

    def test_sentence_builder_integration(self):
        import sentence_builder
        from semantic_attractors import SemanticAttractorBank
        
        bank_path = Path("local_ai/semantic_bank.npz")
        if not bank_path.exists():
            self.skipTest("semantic_bank.npz not found")
            
        builder = sentence_builder.SemanticSentenceBuilder.from_bank_path(bank_path)
        # This will trigger reranking internally
        results = builder.build("cat animal gentle", n=5)
        self.assertTrue(len(results) > 0)
        # Ensure we don't return the bad one at top
        for r in results[:2]:
            self.assertNotEqual(r.sentence, "A cat gentles an animal.")

if __name__ == "__main__":
    unittest.main()
