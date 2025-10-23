from typing import Any, Callable

# Import each tool you want to expose
from .calculator import calculator
from .get_weather import get_weather
from .translator import translator
from .web_reader import web_reader
from .file_system_search import file_system_search

# Put them all into the TOOLS list
TOOLS: list[Callable[..., Any]] = [
    calculator,
    get_weather,
    translator,
    web_reader,
    file_system_search
]
