"""Tests for shared local-agent construction and model routing."""

import unittest
from unittest.mock import Mock

from university_agent.local_agent import create_ollama_agent, select_ollama_model


class LocalAgentConstructionTests(unittest.TestCase):
    def test_automatic_model_routing_is_shared(self):
        self.assertEqual(
            select_ollama_model("Busca en mis apuntes información sobre redes"),
            "llama3.2:3b",
        )
        self.assertEqual(
            select_ollama_model("¿Qué tengo que hacer esta semana?"),
            "qwen3:14b",
        )

    def test_explicit_model_override_wins(self):
        self.assertEqual(
            select_ollama_model(
                "Busca en mis apuntes información sobre redes",
                "fictional-model",
            ),
            "fictional-model",
        )

    def test_factory_receives_existing_backend_dependencies(self):
        agent_factory = Mock()
        client = Mock()
        connector = Mock()
        source = Mock()

        agent, model = create_ollama_agent(
            query="Fictional generic question",
            client=client,
            connector=connector,
            materials_source=source,
            timezone_name="Europe/Madrid",
            num_predict=256,
            agent_factory=agent_factory,
        )

        self.assertIs(agent, agent_factory.return_value)
        self.assertEqual(model, "qwen3:14b")
        agent_factory.assert_called_once_with(
            client=client,
            connector=connector,
            materials_source=source,
            timezone_name="Europe/Madrid",
            model="qwen3:14b",
            num_predict=256,
        )

    def test_unset_generation_budget_is_not_forwarded(self):
        agent_factory = Mock()

        create_ollama_agent(
            query="Fictional question",
            client=Mock(),
            connector=Mock(),
            materials_source=Mock(),
            timezone_name="Europe/Madrid",
            agent_factory=agent_factory,
        )

        self.assertNotIn("num_predict", agent_factory.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
