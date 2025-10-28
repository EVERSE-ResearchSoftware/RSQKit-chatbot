# provider_persistence.py - Add this new module

import streamlit as st
import json
from typing import Dict, Any, Optional
from pathlib import Path
import threading
import time


class ProviderPersistence:
    """
    Multi-layered provider persistence system with:
    1. Browser-like session persistence (primary)
    2. In-memory global state (fallback)
    3. Optional file persistence (enterprise)
    """

    # Global state (survives between Streamlit reruns but not process restarts)
    _global_state: Dict[str, str] = {}
    _lock = threading.Lock()

    # File persistence settings
    _persistence_file = Path(".streamlit_provider_state.json")
    _last_save_time = 0
    _save_delay = 2.0  # Batch writes to reduce I/O

    @classmethod
    def get_provider(cls, page_key: str) -> Optional[str]:
        """
        Get provider for page with multi-layer fallback:
        1. Streamlit session state (browser-session persistence)
        2. Global memory state (process persistence)
        3. File persistence (cross-session persistence)
        """
        provider_key = f"provider_{page_key}"

        # Layer 1: Check Streamlit session state first
        if provider_key in st.session_state:
            provider = st.session_state[provider_key]
            if provider and provider.strip():
                # Sync to global state for consistency
                cls._set_global_state(page_key, provider)
                return provider

        # Layer 2: Check global memory state
        with cls._lock:
            if page_key in cls._global_state:
                provider = cls._global_state[page_key]
                if provider and provider.strip():
                    # Restore to session state
                    st.session_state[provider_key] = provider
                    return provider

        # Layer 3: Check file persistence (optional)
        provider = cls._load_from_file(page_key)
        if provider:
            # Restore to both session state and global state
            st.session_state[provider_key] = provider
            cls._set_global_state(page_key, provider)
            return provider

        return None

    @classmethod
    def set_provider(cls, page_key: str, provider: str):
        """
        Set provider with multi-layer persistence
        """
        provider_key = f"provider_{page_key}"

        # Set in session state (immediate)
        st.session_state[provider_key] = provider

        # Set in global state (process-wide)
        cls._set_global_state(page_key, provider)

        # Schedule file save (batched for performance)
        cls._schedule_file_save(page_key, provider)

    @classmethod
    def _set_global_state(cls, page_key: str, provider: str):
        """Thread-safe global state update"""
        with cls._lock:
            cls._global_state[page_key] = provider

    @classmethod
    def _schedule_file_save(cls, page_key: str, provider: str):
        """
        Schedule a file save with batching to reduce I/O.
        Only saves if enough time has passed since last save.
        """
        current_time = time.time()

        # Update global state immediately
        cls._set_global_state(page_key, provider)

        # Only save to file if enough time has passed (batching)
        if current_time - cls._last_save_time > cls._save_delay:
            cls._save_to_file()
            cls._last_save_time = current_time

    @classmethod
    def _load_from_file(cls, page_key: str) -> Optional[str]:
        """Load provider from file persistence"""
        try:
            if cls._persistence_file.exists():
                with open(cls._persistence_file, "r") as f:
                    data = json.load(f)
                    return data.get(page_key)
        except (json.JSONDecodeError, FileNotFoundError, PermissionError):
            # Gracefully handle file issues
            pass
        return None

    @classmethod
    def _save_to_file(cls):
        """Save current global state to file (batched operation)"""
        try:
            with cls._lock:
                # Only save non-empty providers
                data_to_save = {
                    page: provider
                    for page, provider in cls._global_state.items()
                    if provider and provider.strip()
                }

            if data_to_save:  # Only write if there's actual data
                with open(cls._persistence_file, "w") as f:
                    json.dump(data_to_save, f, indent=2)
        except (PermissionError, OSError):
            # Gracefully handle file write issues
            pass

    @classmethod
    def clear_all(cls):
        """Clear all persistence layers (useful for testing/reset)"""
        # Clear global state
        with cls._lock:
            cls._global_state.clear()

        # Clear file
        try:
            if cls._persistence_file.exists():
                cls._persistence_file.unlink()
        except (PermissionError, OSError):
            pass

    @classmethod
    def get_all_providers(cls) -> Dict[str, str]:
        """Get all stored providers (useful for debugging)"""
        with cls._lock:
            return cls._global_state.copy()
