"""Auditable constrained selection research, isolated from production ranking."""

from .contracts import load_protocol, validate_protocol
from .solver import solve_pool

__all__ = ["load_protocol", "validate_protocol", "solve_pool"]
