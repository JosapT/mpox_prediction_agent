#!/usr/bin/env python3
"""CLI for the monkeypox prediction agent."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from agent.llm import LLMError
from agent.mpox_agent import (
    CHAT_COMMANDS,
    MpoxAgent,
    format_agent_response,
    format_chat_response,
    print_chat_help,
    print_chat_welcome,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Chat with the monkeypox screening agent. Describe symptoms for a "
            "prediction, then ask follow-up questions in the same session."
        )
    )
    parser.add_argument(
        "symptoms",
        nargs="?",
        help="Optional first message. If omitted, an interactive chat starts.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to best_mpox_model.pkl",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single assessment and exit instead of staying in chat mode.",
    )
    return parser


def handle_chat_command(command: str, session) -> bool:
    """Handle built-in chat commands. Returns True if the chat loop should exit."""
    if command in {"quit", "exit", "q"}:
        print("Goodbye.")
        return True

    if command in {"clear", "reset"}:
        session.clear()
        print("Conversation cleared. Describe your symptoms whenever you're ready.\n")
        return False

    if command in {"help", "h"}:
        print_chat_help()
        return False

    return False


def run_chat_loop(session) -> int:
    print_chat_welcome()

    while True:
        try:
            user_query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not user_query:
            continue

        lower = user_query.lower()
        if lower in CHAT_COMMANDS:
            if handle_chat_command(lower, session):
                return 0
            continue

        try:
            response = session.chat(user_query)
        except (LLMError, FileNotFoundError, ValueError) as exc:
            print(f"\nAssistant: Sorry, I ran into an error: {exc}\n")
            continue

        print(f"\nAssistant:\n{format_chat_response(response)}\n")
        print("—" * 48 + "\n")


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        agent = MpoxAgent(model_path=args.model_path)
        session = agent.start_chat()
    except (LLMError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.symptoms:
        try:
            response = session.chat(args.symptoms)
        except (LLMError, FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        output = (
            format_agent_response(response)
            if args.once
            else format_chat_response(response)
        )
        print(output)

        if args.once:
            return 0

        print("\n" + "—" * 48 + "\n")

    return run_chat_loop(session)


if __name__ == "__main__":
    raise SystemExit(main())
