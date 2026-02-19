"""
Global constants for AquaWatch.
"""

RISK_LEVELS = {
    "SAFE":     {"min": 0,  "max": 25,  "color": "#2ecc71", "emoji": "✅", "label": "Safe"},
    "LOW":      {"min": 25, "max": 50,  "color": "#f1c40f", "emoji": "⚠️", "label": "Low Risk"},
    "WARNING":  {"min": 50, "max": 75,  "color": "#e67e22", "emoji": "🟠", "label": "Warning"},
    "CRITICAL": {"min": 75, "max": 100, "color": "#e74c3c", "emoji": "🔴", "label": "Critical"},
}

WHO_CYANO_THRESHOLDS = [
    {"label": "WHO Low",       "cells": 20_000,     "color": "#2ecc71",
     "description": "Relatively low probability of adverse health effects."},
    {"label": "WHO Moderate",  "cells": 100_000,    "color": "#f1c40f",
     "description": "Moderate probability of adverse health effects."},
    {"label": "WHO High",      "cells": 10_000_000, "color": "#e74c3c",
     "description": "High probability of adverse health effects during recreational exposure."},
]
