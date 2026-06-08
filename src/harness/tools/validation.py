from __future__ import annotations

from dataclasses import dataclass

import jsonschema
from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ValidationError:
    message: str
    path: str  # JSON-pointer-ish; e.g. "args.expression"

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def validate(args: dict, schema: dict) -> list[ValidationError]:
    """Return a list of validation errors. Empty list == valid."""
    validator = Draft202012Validator(schema)
    errors: list[ValidationError] = []
    for err in validator.iter_errors(args):
        path = "args" + "".join(f".{p}" for p in err.absolute_path)
        errors.append(ValidationError(message=err.message, path=path))
    return errors