"""Count real calls from pipeline operations into a dependency under study."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class CallTrace:
    def __init__(self, dependency: str | None) -> None:
        self.dependency = dependency
        self.counts: Counter[str] = Counter()

    def call(self, operation: str, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.dependency is None or operation.split(".", 1)[0] == self.dependency:
            self.counts[operation] += 1
        return function(*args, **kwargs)
