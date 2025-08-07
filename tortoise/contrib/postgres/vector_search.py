from tortoise.queryset import QuerySet

from enum import Enum
from typing import List, Optional
from tortoise.contrib.postgres.fields import VectorField
from tortoise.expressions import RawSQL

class SearchType(str, Enum):
    """
    Defines the available pgvector similarity search types.
    """
    L2_DISTANCE = '<->'
    COSINE_SIMILARITY = '<=>'
    INNER_PRODUCT = '<#>'

class VectorQuerySet(QuerySet):
    """
    A custom QuerySet to add pgvector-specific search capabilities.
    """
    def order_by_similarity(
        self,
        query_vector: List[float],
        search_type: SearchType,
        vector_field_name: str = None
    ) -> 'VectorQuerySet':
        """
        Allows for L1, L2, cosine, and inner product searches on a specified VectorField.

        Args:
            query_vector (List[float]): The vector to search against.
            search_type (str): The type of similarity search to perform (e.g., 'L2_DISTANCE').
            vector_field_name (Optional[str]): The name of the VectorField to search on.
                                               If None, the first VectorField found will be used.
        """
        if vector_field_name:
            vector_field = self.model._meta.fields_map.get(vector_field_name)
            if not isinstance(vector_field, VectorField):
                raise ValueError(f"Field '{vector_field_name}' is not a VectorField.")

        if not vector_field:
            raise RuntimeError(f"Model '{self.model.__name__}' does not contain a VectorField '{vector_field_name}'.")

        embedding_column_name = vector_field.model_field_name
        sql_symbol = SearchType[search_type].value

        vector_string = f"[{','.join(str(f) for f in query_vector)}]"
        raw_sql_expression = f'"{embedding_column_name}" {sql_symbol} \'{vector_string}\''

        annotated_queryset = self.annotate(
            distance=RawSQL(raw_sql_expression)
        )

        return annotated_queryset.order_by('distance')

