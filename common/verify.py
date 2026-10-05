"""Small helper for the exercise check cells.

A check that passes prints a line with a tick; one that fails raises an
AssertionError with a message that tells you what is wrong. If the check cannot even run (a variable
does not exist, is still None, has the wrong type) it says so in plain words, instead of showing a technical error.
"""
from types import TracebackType
from typing import Any, Dict, Optional, Type

import numpy as np


def check(condition: Any, ok: str, error: str) -> None:
    """Check one condition of an exercise.

    Args:
        condition: anything that can be converted to a bool (a bool, a numpy bool, a non-empty value...).
        ok: message printed, with a tick, when the condition holds.
        error: message of the AssertionError raised when it does not: it should say what is wrong or where to look.

    Raises:
        AssertionError: if the condition is false.
    """
    if bool(condition):
        print(f"✔ {ok}")
    else:
        raise AssertionError(f"✘ {error}")


def close(a: Any, b: Any, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    """`np.allclose` that also accepts scalars, lists and pandas objects, and returns a plain bool.

    Args:
        a: first value (anything numeric that numpy can turn into an array of floats).
        b: second value, compared with `a` element by element.
        rtol: relative tolerance.
        atol: absolute tolerance.
    """
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=rtol, atol=atol))


def done() -> None:
    """Call it at the end of a check cell: it confirms that all the checks passed."""
    print("\nExercise passed. Compare your code with the solution and read 'What you learned'.")


# Failed runs seen so far in this kernel, per hint text: the hint is shown from the second failure.
_FAILED_ATTEMPTS: Dict[str, int] = {}


class verify:
    """Context manager that wraps a check cell: it turns technical errors into an understandable message.

    Usage:  with verify(hint="..."):
                check(...)

    The hint stays hidden on the first failed run (the starter code always fails: that is expected) and appears
    from the second failed run of the same check cell, so you get one honest attempt before being helped.

    Args:
        hint: optional help text shown under the failure message from the second failed run onwards.
    """

    def __init__(self, hint: Optional[str] = None) -> None:
        self.hint = hint

    def __enter__(self) -> "verify":
        return self

    def __exit__(self, kind: Optional[Type[BaseException]], value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> bool:
        """Re-raise failures with a readable message; return False so that a passing block continues normally."""
        if kind is not None and self.hint and issubclass(kind, (AssertionError, NameError, TypeError, ValueError, KeyError, AttributeError)):
            attempts = _FAILED_ATTEMPTS[self.hint] = _FAILED_ATTEMPTS.get(self.hint, 0) + 1
            if attempts >= 2 and issubclass(kind, AssertionError):
                raise AssertionError(f"{value}\n\nHint: {self.hint}") from None
        if kind is None or issubclass(kind, AssertionError):
            return False                                    # all fine, or a failed check with its own message
        if issubclass(kind, NameError):
            raise AssertionError(f"✘ A variable is missing ({value}): did you run the 'Your work' cells and give every "
                                 "result the name required by the Task?") from None
        raise AssertionError(f"✘ A check cannot run ({kind.__name__}: {value}). Usually it means that a "
                             "result is still None, or has the wrong type or shape: reread the Task.") from None
