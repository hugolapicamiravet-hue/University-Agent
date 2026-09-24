"""Regression test for literal terms in lexical material queries."""

import unittest

from university_agent.agent_tools import SYSTEM_INSTRUCTIONS


class AgentLexicalInstructionTests(unittest.TestCase):
    def test_distinctive_query_terms_are_not_translated(self):
        normalized = " ".join(SYSTEM_INSTRUCTIONS.split())

        self.assertIn(
            "Preserve distinctive terms from the user verbatim in lexical "
            "material queries; do not translate them.",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
