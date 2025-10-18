"""
Example tool implementations for the AI Assistant tool system.

This module provides basic example tools that demonstrate the tool system's
capabilities and serve as references for creating new tools.
"""

import math
import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Any, List
from .base import BaseTool, ToolExecutionError


class CalculatorTool(BaseTool):
    """Perform mathematical calculations"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations and conversions"

    @property
    def keywords(self) -> List[str]:
        return [
            "calculate",
            "math",
            "equation",
            "convert",
            "sum",
            "multiply",
            "divide",
            "add",
            "subtract",
            "percentage",
            "sqrt",
            "power",
        ]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "expression": {
                "type": str,
                "description": "Mathematical expression to evaluate",
                "required": True,
            }
        }

    async def execute(self, expression: str) -> float:
        """Evaluate mathematical expression safely"""
        # Safe evaluation - only allow math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("_")
        }

        # Remove dangerous builtins
        allowed_names["__builtins__"] = {}

        try:
            # Use ast.literal_eval as a safer alternative to eval for simple expressions
            import ast

            try:
                # Try literal_eval first for safety
                result = ast.literal_eval(expression)
                # Ensure the result is a number
                if not isinstance(result, (int, float)):
                    raise ValueError("Expression must evaluate to a number")
            except (ValueError, SyntaxError):
                # Fall back to eval with restricted namespace for complex math expressions
                result = eval(expression, allowed_names)  # nosec B307
                if not isinstance(result, (int, float)):
                    raise ValueError("Expression must evaluate to a number")
            if not isinstance(result, (int, float)):
                raise ValueError("Expression must evaluate to a number")
            return float(result)
        except Exception as e:
            raise ToolExecutionError(f"Failed to evaluate expression: {e}")


class TimeTool(BaseTool):
    """Get current time and date information"""

    @property
    def name(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Get current time, date, and timezone information"

    @property
    def keywords(self) -> List[str]:
        return ["time", "date", "now", "current", "timezone", "clock", "today"]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "timezone": {
                "type": str,
                "description": "Timezone name (e.g., 'UTC', 'US/Eastern')",
                "required": False,
                "default": "UTC",
            },
            "format": {
                "type": str,
                "description": "Output format (e.g., 'iso', 'human')",
                "required": False,
                "default": "human",
            },
        }

    async def execute(
        self, timezone: str = "UTC", format: str = "human"
    ) -> Dict[str, Any]:
        """Get current time information"""
        try:
            # Get timezone using zoneinfo (built-in Python 3.9+)
            tz = ZoneInfo(timezone)
            current_time = datetime.datetime.now(tz)

            if format == "iso":
                time_str = current_time.isoformat()
            else:
                time_str = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

            return {
                "time": time_str,
                "timezone": timezone,
                "timestamp": current_time.timestamp(),
                "utc_offset": (
                    current_time.utcoffset().total_seconds() / 3600
                    if current_time.utcoffset()
                    else 0
                ),
            }
        except Exception as e:
            raise ToolExecutionError(f"Failed to get time: {e}")


class EchoTool(BaseTool):
    """Simple echo tool for testing"""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input text (for testing purposes)"

    @property
    def keywords(self) -> List[str]:
        return ["echo", "repeat", "test"]

    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "text": {"type": str, "description": "Text to echo back", "required": True}
        }

    async def execute(self, text: str) -> str:
        """Echo back the input text"""
        return f"Echo: {text}"


def initialize_default_tools():
    """Initialize and register default example tools"""
    from .registry import tool_registry

    # Register example tools
    tool_registry.register(CalculatorTool(), category="utility")
    tool_registry.register(TimeTool(), category="utility")
    tool_registry.register(EchoTool(), category="testing")

    return tool_registry
