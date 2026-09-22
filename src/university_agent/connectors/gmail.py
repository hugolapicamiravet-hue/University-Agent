"""Read-only Gmail connector using OAuth 2.0."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = [GMAIL_READONLY_SCOPE]


class GmailMessageSummary(TypedDict):
    """Header fields returned for a Gmail message."""

    sender: str
    subject: str
    date: str


class GmailConnector:
    """Authenticate with Gmail and read recent message headers."""

    def __init__(
        self,
        credentials_path: str | Path = "secrets/credentials.json",
        token_path: str | Path = "secrets/token.json",
    ) -> None:
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self._service: Any | None = None

    def authenticate(self) -> None:
        """Authenticate the user and prepare the read-only Gmail service."""
        credentials: Credentials | None = None

        if self.token_path.exists():
            credentials = Credentials.from_authorized_user_file(
                self.token_path,
                SCOPES,
            )

        if not credentials or not credentials.valid:
            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                if not self.credentials_path.is_file():
                    raise FileNotFoundError(
                        f"Gmail OAuth credentials not found: {self.credentials_path}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path,
                    SCOPES,
                )
                credentials = flow.run_local_server(port=0)

            self._save_token(credentials)

        self._service = build(
            "gmail",
            "v1",
            credentials=credentials,
            cache_discovery=False,
        )

    def fetch_recent_messages(self) -> list[GmailMessageSummary]:
        """Return sender, subject, and date for the 10 most recent messages."""
        if self._service is None:
            self.authenticate()

        response = (
            self._service.users()
            .messages()
            .list(userId="me", maxResults=10)
            .execute()
        )

        summaries: list[GmailMessageSummary] = []
        for message in response.get("messages", []):
            details = (
                self._service.users()
                .messages()
                .get(
                    userId="me",
                    id=message["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                )
                .execute()
            )
            headers = {
                header["name"].casefold(): header["value"]
                for header in details.get("payload", {}).get("headers", [])
            }
            summaries.append(
                {
                    "sender": headers.get("from", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
                }
            )

        return summaries

    def _save_token(self, credentials: Credentials) -> None:
        """Store OAuth tokens locally with owner-only file permissions."""
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.token_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
            token_file.write(credentials.to_json())
