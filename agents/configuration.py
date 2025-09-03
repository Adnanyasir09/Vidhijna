# configuration.py

import os
from dataclasses import dataclass, fields
from typing import Optional, Any, Dict
from enum import Enum
from langchain_core.runnables import RunnableConfig

# ----------------------------
# Helper: Project root (top-level project folder)
# ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------
# Helper: Normalize paths relative to PROJECT_ROOT
# ----------------------------
def normalize_path(path: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path

# ----------------------------
# Search API Enum
# ----------------------------
class SearchAPI(str, Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"

# ----------------------------
# Configuration Dataclass
# ----------------------------
@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""

    # ----------------------------
    # Research loop controls
    # ----------------------------
    max_web_research_loops: int = 3
    max_vector_store_research_loops: int = 1
    search_depth: int = 3

    # ----------------------------
    # LLM config
    # ----------------------------
    local_llm: str = "gemma3:1b"
    ollama_base_url: str = "http://localhost:11434/"

    # ----------------------------
    # Search provider
    # ----------------------------
    search_api: SearchAPI = SearchAPI.DUCKDUCKGO
    fetch_full_page: bool = False

    # ----------------------------
    # Vector store paths (absolute)
    # ----------------------------
    laws_faiss_path: str = normalize_path("commercial_laws_index")
    cases_faiss_path: str = normalize_path("cases_index")
    commercial_laws_pdf: str = normalize_path(os.path.join("commercial_cases", "merged_output_cases.pdf"))

    # ----------------------------
    # Developer toggles
    # ----------------------------
    dev_mode: bool = True
    debug_mode: bool = True
    log_dir: str = normalize_path("logs")

    # ----------------------------
    # Ensure directories exist
    # ----------------------------
    def __post_init__(self):
        os.makedirs(self.laws_faiss_path, exist_ok=True)
        os.makedirs(self.cases_faiss_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.commercial_laws_pdf), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # ----------------------------
    # Create Configuration from RunnableConfig
    # ----------------------------
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        values: Dict[str, Any] = {}

        for f in fields(cls):
            value = configurable.get(f.name, getattr(cls, f.name, None))
            # Normalize paths if relevant
            if f.name in ["laws_faiss_path", "cases_faiss_path", "commercial_laws_pdf", "log_dir"]:
                value = normalize_path(value)
            values[f.name] = value

        return cls(**values)

    # ----------------------------
    # Convert configuration to dict
    # ----------------------------
    def to_dict(self) -> dict:
        """Return the configuration as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
