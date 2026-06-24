"""
Service layer for the web-enabled DBN pipeline corrosion assessment platform.

The PySMILE license is intentionally loaded from a local, non-versioned
environment variables or a local, non-versioned ``pysmile_license.py`` file.
Do not commit personal BayesFusion license data to public repositories.
"""

import base64
import os
from threading import Lock
from typing import Dict

import pysmile


_LICENSE_LOCK = Lock()
_LICENSE_INITIALIZED = False


def _initialize_pysmile_license():
    """Initialize PySMILE from environment variables or a local license file."""
    global _LICENSE_INITIALIZED
    with _LICENSE_LOCK:
        if _LICENSE_INITIALIZED:
            return

        license_b64 = os.getenv("PYSMILE_LICENSE_B64")
        license_text = os.getenv("PYSMILE_LICENSE")
        license_key_text = os.getenv("PYSMILE_LICENSE_KEY")
        license_key = (
            [int(value) for value in license_key_text.split(",") if value.strip()]
            if license_key_text
            else None
        )
        if license_b64:
            if license_key:
                pysmile.License(base64.b64decode(license_b64), license_key)
            else:
                pysmile.License(base64.b64decode(license_b64))
            _LICENSE_INITIALIZED = True
            return
        if license_text:
            if license_key:
                pysmile.License(license_text.encode("utf-8"), license_key)
            else:
                pysmile.License(license_text.encode("utf-8"))
            _LICENSE_INITIALIZED = True
            return

        try:
            from pysmile_license import apply_pysmile_license
        except ImportError as exc:
            raise RuntimeError(
                "PySMILE license is not configured. Set PYSMILE_LICENSE_B64 "
                "and PYSMILE_LICENSE_KEY, or set PYSMILE_LICENSE, or copy "
                "pysmile_license.example.py to pysmile_license.py and add "
                "your own BayesFusion license."
            ) from exc

        apply_pysmile_license(pysmile)
        _LICENSE_INITIALIZED = True


class PipelineCorrosionDBN:
    """Thin wrapper around the GeNIe/PySMILE DBN model."""

    def __init__(self, model_path: str):
        _initialize_pysmile_license()
        self.net = pysmile.Network()
        self.net.read_file(model_path)
        self.target_node = "E1"

    def set_current_conditions(self, conditions: Dict[str, str]):
        """Set evidence for model nodes at the current time slice."""
        invalid_conditions = []
        for node_id, state in conditions.items():
            try:
                self.net.set_evidence(node_id, state)
            except Exception as exc:
                invalid_conditions.append(f"{node_id}={state} ({exc})")

        if invalid_conditions:
            details = "; ".join(invalid_conditions)
            raise ValueError(f"Invalid evidence setting(s): {details}")

    def predict_corrosion(self, time_slice: int = None):
        """
        Run DBN inference and return posterior probabilities for every time slice.

        Returns:
            dict: {
                "t=0": {"Low": p_low, "Moderate": p_moderate, "High": p_high},
                ...
            }
        """
        self.net.update_beliefs()

        outcomes = [
            self.net.get_outcome_id(self.target_node, index)
            for index in range(self.net.get_outcome_count(self.target_node))
        ]

        if time_slice is not None:
            beliefs = self.net.get_node_value(self.target_node)
            outcome_count = len(outcomes)
            start_index = time_slice * outcome_count
            end_index = start_index + outcome_count
            return {
                outcome: float(probability)
                for outcome, probability in zip(outcomes, beliefs[start_index:end_index])
            }

        beliefs = self.net.get_node_value(self.target_node)
        outcome_count = len(outcomes)
        slice_count = self.net.get_slice_count()
        results = {}
        for slice_index in range(slice_count):
            start_index = slice_index * outcome_count
            end_index = start_index + outcome_count
            slice_beliefs = beliefs[start_index:end_index]
            results[f"t={slice_index}"] = {
                outcome: float(probability)
                for outcome, probability in zip(outcomes, slice_beliefs)
            }
        return results

    def scenario_analysis(self, base_conditions, scenarios, target_slice=10):
        """Evaluate terminal corrosion probabilities for named scenario changes."""
        results = {}
        for scenario_name, modifications in scenarios.items():
            self.net.clear_all_evidence()
            scenario_conditions = dict(base_conditions)
            scenario_conditions.update(modifications)
            self.set_current_conditions(scenario_conditions)
            prediction = self.predict_corrosion(time_slice=target_slice)
            results[scenario_name] = prediction
        self.net.clear_all_evidence()
        return results

    def get_model_info(self):
        """Return basic model metadata for diagnostics."""
        return {
            "node_count": self.net.get_node_count(),
            "slice_count": self.net.get_slice_count(),
            "target_node": self.target_node,
            "target_outcomes": [
                self.net.get_outcome_id(self.target_node, index)
                for index in range(self.net.get_outcome_count(self.target_node))
            ],
        }
