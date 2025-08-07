from __future__ import annotations

from typing import Any, List

from tortoise.fields import Field

import json


class TSVectorField(Field):
    SQL_TYPE = "TSVECTOR"


class ArrayField(Field, list):  # type: ignore
    def __init__(self, element_type: str = "int", **kwargs: Any):
        super().__init__(**kwargs)
        self.element_type = element_type.upper()

    @property
    def SQL_TYPE(self) -> str:  # type: ignore
        return f"{self.element_type}[]"


class VectorField(Field[List[float]], list):
    """
    Implementation of pgvector's vector fields.

    args:
    - dimensions: max=2,000 (pgvector's upper limit), min=1 (pgvector's lower limit)
    """

    def __init__(self, dimensions: int, null: bool | None = None, **kwargs: Any) -> None:
        if not (1 <= dimensions <= 2000):
            raise ValueError("VectorField dimensions must be between 1 and 2000.")
        super().__init__(null=null, **kwargs)
        self.dimensions = dimensions

    # I'll very likely want to implement a constraints property.
    # @property
    # def constraints(self) -> dict:
    #   pass
    allows_generated = False

    class _db_postgres:
        SQL_TYPE = "vector"
        def get_db_type(self, field_instance: "VectorField", is_generated: bool) -> str:
            return f"VECTOR({field_instance.dimensions})"

    def to_python_value(self, value) -> Optional[List[float]]:
        if value is None:
            return None
        elif isinstance(value, list):
            return value
        elif isinstance(value, (str, bytes, bytearray)):
            return json.loads(value)
        else:
            raise TypeError(f"Cannot deserialize value of type {type(value)}")
        raise TypeError(f"Cannot deserialize value of type {type(value)}")

    def to_db_value(self, value: Optional[List[float]], instance: Any) -> Optional[str]:
        if value is None:
            return None
        if len(value) != self.dimensions:
            raise ValueError(f"Vector length ({len(value)}) must match field dimensions ({self.dimensions}).")
        return f"[{', '.join(str(v) for v in value)}]"
