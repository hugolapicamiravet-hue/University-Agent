"""Regression tests for provider-shared agent instructions."""

import unittest

from university_agent.agent_tools import SYSTEM_INSTRUCTIONS


class AgentInstructionTests(unittest.TestCase):
    def test_unspecified_material_course_searches_all_courses(self):
        normalized = " ".join(SYSTEM_INSTRUCTIONS.split())

        self.assertIn(
            "local-material questions without a course name, search all "
            "configured courses by passing null",
            normalized,
        )
        self.assertNotIn("when a course is ambiguous", normalized)


if __name__ == "__main__":
    unittest.main()
