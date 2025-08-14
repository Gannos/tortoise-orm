from tortoise.manager import Manager
from tortoise.contrib.postgres.vector_search import VectorQuerySet

class VectorManager(Manager):
    def get_queryset(self) -> VectorQuerySet:
        return VectorQuerySet(self._model)

    def all(self) -> 'VectorQuerySet':
        # Overriding the 'all' method is the key.
        # It must return a QuerySet, not try to call 'all' on it,
        # which would lead to recursion.
        return self.get_queryset()
