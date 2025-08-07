from tortoise import Model, fields
from tortoise.contrib.postgres.fields import ArrayField, VectorField
from tortoise.contrib.postgres.vector_manager import VectorManager


class ArrayFields(Model):
    id = fields.IntField(primary_key=True)
    array = ArrayField()
    array_null = ArrayField(null=True)
    array_str = ArrayField(element_type="varchar(1)", null=True)
    array_smallint = ArrayField(element_type="smallint", null=True)

class VectorFields(Model):
    id = fields.IntField(primary_key=True)
    embedding_1536 = VectorField(dimensions=1536, null=True)
    embedding_3 = VectorField(dimensions=3, null=True)

    class Meta:
        manager = VectorManager()
