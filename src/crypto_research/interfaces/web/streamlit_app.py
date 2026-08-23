"""Streamlit entrypoint kept outside a reserved sibling ``pages`` directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from crypto_research.interfaces.web.app import main
from crypto_research.interfaces.web.launcher import streamlit_version_gap_message

_gap_message = streamlit_version_gap_message()
if _gap_message is not None:
    st.error(_gap_message)
    st.stop()

main()
