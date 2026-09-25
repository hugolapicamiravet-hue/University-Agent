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

    def test_generic_questions_avoid_material_search_and_locations_are_concise(self):
        normalized = " ".join(SYSTEM_INSTRUCTIONS.split())

        self.assertIn(
            "Do not search local materials for a generic academic question "
            "that does not explicitly refer to the user's notes, materials, "
            "or local sources",
            normalized,
        )
        self.assertIn(
            "When the user asks where information appears, answer concisely",
            normalized,
        )

    def test_material_results_have_host_sources_and_safe_empty_behavior(self):
        normalized = " ".join(SYSTEM_INSTRUCTIONS.split())

        self.assertIn(
            "Do not create a separate sources section because the host "
            "appends deterministic source provenance",
            normalized,
        )
        self.assertIn(
            "If a tool returns no results, state clearly that no matching "
            "evidence was found",
            normalized,
        )
        self.assertIn(
            "do not present an answer as grounded in those sources",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()
