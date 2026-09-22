"""Manually verify read-only Gmail OAuth and recent-message access."""

from university_agent.connectors.gmail import GmailConnector


def main() -> None:
    connector = GmailConnector()

    for message in connector.fetch_recent_messages():
        print(f"From: {message['sender']}")
        print(f"Subject: {message['subject']}")
        print(f"Date: {message['date']}")
        print()


if __name__ == "__main__":
    main()
