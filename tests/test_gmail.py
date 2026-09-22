"""Isolated tests: no OAuth, credential files, or Gmail network access."""

import unittest
from unittest.mock import Mock, call, patch

from university_agent.connectors.gmail import GmailConnector, SCOPES


class GmailConnectorTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.messages = self.service.users.return_value.messages.return_value
        self.connector = GmailConnector(service=self.service)
        self.auth = self.enterContext(
            patch.object(
                GmailConnector, "authenticate",
                side_effect=AssertionError("OAuth forbidden"),
            )
        )
        self.messages.get.return_value.execute.return_value = {
            "id": "m1", "threadId": "t1", "snippet": "Preview",
            "payload": {"headers": [
                {"name": "fRoM", "value": "Teacher <teacher@example.com>"},
                {"name": "TO", "value": "student@example.com"},
                {"name": "subject", "value": "Notice"},
                {"name": "Date", "value": "Tue, 22 Sep 2026 10:00:00 +0200"},
            ]},
        }

    def test_metadata_mapping_and_exact_request(self):
        result = self.connector.get_message("m1")
        self.messages.get.assert_called_once_with(
            userId="me", id="m1", format="metadata",
            metadataHeaders=["From", "To", "Subject", "Date"],
        )
        self.assertEqual(result, {
            "id": "m1", "thread_id": "t1", "snippet": "Preview",
            "sender": "Teacher <teacher@example.com>",
            "recipients": "student@example.com", "subject": "Notice",
            "date": "Tue, 22 Sep 2026 10:00:00 +0200",
        })
        self.auth.assert_not_called()

    def test_missing_headers_and_snippet(self):
        for extra in ({}, {"payload": {}}, {"payload": {"headers": []}}):
            with self.subTest(extra=extra):
                self.messages.get.return_value.execute.return_value = {
                    "id": "m1", "threadId": "t1", **extra,
                }
                self.assertEqual(self.connector.get_message("m1"), {
                    "id": "m1", "thread_id": "t1", "sender": "",
                    "recipients": "", "subject": "", "date": "", "snippet": "",
                })

    def test_query_and_limit_forwarding_and_get_reuse(self):
        query = 'from:teacher@example.com subject:"exam date"'
        self.messages.list.return_value.execute.return_value = {
            "messages": [{"id": "m2"}, {"id": "m1"}], "nextPageToken": "next",
        }
        results = [{"id": "m2"}, {"id": "m1"}]
        with patch.object(self.connector, "get_message", side_effect=results) as get:
            self.assertEqual(self.connector.search_messages(query, 2), results)
            self.assertEqual(get.call_args_list, [call("m2"), call("m1")])
        self.messages.list.assert_called_once_with(userId="me", q=query, maxResults=2)
        self.auth.assert_not_called()

    def test_empty_search_results(self):
        for response in ({}, {"messages": []}):
            with self.subTest(response=response):
                self.messages.list.return_value.execute.return_value = response
                self.assertEqual(self.connector.search_messages(""), [])
        self.messages.get.assert_not_called()
        self.messages.list.assert_called_with(userId="me", q="", maxResults=10)

    def test_invalid_limits_before_authentication(self):
        connector = GmailConnector()
        for limit in (0, -1, 501):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                connector.search_messages("", limit)
        self.auth.assert_not_called()

    def test_empty_ids_before_authentication(self):
        connector = GmailConnector()
        for message_id in ("", "   "):
            with self.subTest(message_id=message_id), self.assertRaises(ValueError):
                connector.get_message(message_id)
        self.auth.assert_not_called()

    def test_recent_messages_compatibility(self):
        self.messages.list.return_value.execute.return_value = {"messages": [{"id": "m1"}]}
        self.assertEqual(self.connector.fetch_recent_messages(), [{
            "sender": "Teacher <teacher@example.com>", "subject": "Notice",
            "date": "Tue, 22 Sep 2026 10:00:00 +0200",
        }])
        self.messages.list.assert_called_once_with(userId="me", q="", maxResults=10)
        self.auth.assert_not_called()

    def test_api_errors_propagate(self):
        self.messages.get.return_value.execute.side_effect = RuntimeError("API failure")
        with self.assertRaisesRegex(RuntimeError, "API failure"):
            self.connector.get_message("m1")

    def test_scope_unchanged(self):
        self.assertEqual(SCOPES, ["https://www.googleapis.com/auth/gmail.readonly"])


class GmailConnectorAuthenticationTests(unittest.TestCase):
    def test_missing_service_authenticates_once_and_uses_resulting_service(self):
        service = Mock()
        messages = service.users.return_value.messages.return_value
        messages.list.return_value.execute.return_value = {}
        connector = GmailConnector()

        def install_service():
            connector._service = service

        with patch.object(
            connector,
            "authenticate",
            side_effect=install_service,
        ) as authenticate:
            self.assertEqual(connector.search_messages("is:unread", 5), [])

        authenticate.assert_called_once_with()
        messages.list.assert_called_once_with(
            userId="me",
            q="is:unread",
            maxResults=5,
        )


if __name__ == "__main__":
    unittest.main()
