"""Isolated tests: no OAuth, credential files, or Gmail network access."""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

from google.auth.exceptions import RefreshError

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
        self.messages.list.return_value.execute.assert_called_once_with(num_retries=2)
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
            with self.subTest(body_message_id=message_id), self.assertRaises(ValueError):
                connector.get_plain_text_body(message_id)
        self.auth.assert_not_called()

    def test_plain_text_body_is_decoded_from_nested_mime_parts(self):
        encoded = base64.urlsafe_b64encode(
            "Contenido académico ficticio con acentos.".encode()
        ).decode().rstrip("=")
        self.messages.get.return_value.execute.return_value = {
            "payload": {
                "mimeType": "multipart/alternative",
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "headers": [
                            {
                                "name": "Content-Type",
                                "value": "text/plain; charset=utf-8",
                            }
                        ],
                        "body": {"data": encoded},
                    },
                    {
                        "mimeType": "text/html",
                        "body": {"data": "PGI-SFRNTDwvYj4="},
                    },
                ],
            }
        }

        result = self.connector.get_plain_text_body("m1")

        self.assertEqual(result, "Contenido académico ficticio con acentos.")
        self.messages.get.assert_called_once_with(
            userId="me", id="m1", format="full"
        )
        self.messages.get.return_value.execute.assert_called_once_with(num_retries=2)
        self.auth.assert_not_called()

    def test_malformed_or_unsupported_body_is_safely_empty(self):
        payloads = (
            {},
            {"mimeType": "text/plain", "body": None, "headers": None},
            {"mimeType": "multipart/mixed", "parts": None},
            {"mimeType": "text/html", "body": {"data": "PGI-SFRNTDwvYj4="}},
            {"mimeType": "text/plain", "body": {"data": "%%%invalid%%%"}},
            {
                "mimeType": "text/plain",
                "filename": "attachment.txt",
                "body": {"data": "SGVsbG8="},
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                self.messages.get.return_value.execute.return_value = {
                    "payload": payload
                }
                self.assertEqual(self.connector.get_plain_text_body("m1"), "")

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

    def test_revoked_refresh_token_restarts_read_only_oauth_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials_path = root / "credentials.json"
            token_path = root / "token.json"
            credentials_path.write_text("{}", encoding="utf-8")
            token_path.write_text("{}", encoding="utf-8")

            expired = Mock(
                valid=False,
                expired=True,
                refresh_token="fictional-refresh-token",
            )
            expired.refresh.side_effect = RefreshError("revoked")
            renewed = Mock(valid=True)
            flow = Mock()
            flow.run_local_server.return_value = renewed
            service = Mock()
            connector = GmailConnector(
                credentials_path=credentials_path,
                token_path=token_path,
            )

            with (
                patch(
                    "university_agent.connectors.gmail."
                    "Credentials.from_authorized_user_file",
                    return_value=expired,
                ),
                patch(
                    "university_agent.connectors.gmail."
                    "InstalledAppFlow.from_client_secrets_file",
                    return_value=flow,
                ) as create_flow,
                patch.object(connector, "_save_token") as save_token,
                patch(
                    "university_agent.connectors.gmail.build",
                    return_value=service,
                ) as build,
            ):
                connector.authenticate()

        expired.refresh.assert_called_once()
        create_flow.assert_called_once_with(credentials_path, SCOPES)
        flow.run_local_server.assert_called_once_with(port=0)
        save_token.assert_called_once_with(renewed)
        build.assert_called_once_with(
            "gmail",
            "v1",
            credentials=renewed,
            cache_discovery=False,
        )
        self.assertIs(connector._service, service)


if __name__ == "__main__":
    unittest.main()
