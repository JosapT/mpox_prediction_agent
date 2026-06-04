"""Streamlit chat UI for the monkeypox screening agent."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from agent.llm import LLMError, resolve_llm_config
from agent.mpox_agent import MpoxAgent, format_chat_response

load_dotenv()

st.set_page_config(
    page_title="Mpox Screening Chat",
    page_icon="🩺",
    layout="centered",
)

DISCLAIMER = (
    "This tool is for educational screening only and is not a medical diagnosis. "
    "Always consult a qualified healthcare provider for diagnosis and care."
)


@st.cache_resource
def load_agent() -> MpoxAgent:
    return MpoxAgent()


def init_session_state() -> None:
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = load_agent().start_chat()
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = []


def clear_chat() -> None:
    st.session_state.chat_session.clear()
    st.session_state.ui_messages = []


def render_assessment_details(response) -> None:
    prediction = response.prediction
    active_features = [
        key.replace("_", " ")
        for key, value in response.parsed_features.items()
        if key != "systemic_illness" and value == 1
    ]
    systemic = str(response.parsed_features.get("systemic_illness", "none")).replace("_", " ")

    col1, col2, col3 = st.columns(3)
    col1.metric("Result", prediction.label.upper())
    col2.metric("Probability", f"{prediction.probability:.1%}")
    col3.metric("Threshold", f"{prediction.threshold:.2f}")

    with st.expander("Parsed features"):
        if active_features:
            st.write("Symptoms / risk factors:", ", ".join(active_features))
        else:
            st.write("Symptoms / risk factors: none flagged")
        st.write("Systemic illness:", systemic)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("About")
        st.caption(DISCLAIMER)

        try:
            llm_config = resolve_llm_config()
            st.write(f"**LLM:** {llm_config.provider} ({llm_config.model})")
        except LLMError:
            st.error("No LLM backend configured. Set GROQ_API_KEY or run Ollama.")

        st.write("**Model:** SMOTE logistic (best saved classifier)")

        st.divider()
        st.markdown(
            """
**Tips**
- Describe symptoms in plain language.
- Ask follow-ups like *What should I do next?*
- Add new symptoms anytime for an updated assessment.
            """
        )

        if st.button("Clear conversation", use_container_width=True):
            clear_chat()
            st.rerun()


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("Monkeypox Screening Chat")
    st.caption("Describe your symptoms, get a model-based screening estimate, and ask follow-up questions.")

    for message in st.session_state.ui_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("response"):
                response = message["response"]
                if not response.is_follow_up:
                    render_assessment_details(response)

    prompt = st.chat_input("Describe symptoms or ask a follow-up question...")
    if not prompt:
        return

    st.session_state.ui_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your message..."):
            try:
                response = st.session_state.chat_session.chat(prompt)
            except (LLMError, FileNotFoundError, ValueError) as exc:
                error_text = f"Sorry, something went wrong: {exc}"
                st.error(error_text)
                st.session_state.ui_messages.append(
                    {"role": "assistant", "content": error_text}
                )
                return

        reply = format_chat_response(response)
        st.markdown(reply)
        if not response.is_follow_up:
            render_assessment_details(response)

    st.session_state.ui_messages.append(
        {
            "role": "assistant",
            "content": reply,
            "response": response,
        }
    )


if __name__ == "__main__":
    main()
