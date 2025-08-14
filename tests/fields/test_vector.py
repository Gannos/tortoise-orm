from tortoise.contrib import test
# tests/fields/test_vector.py

import pytest
import os # You might not need this anymore if db_url is not in your test file.
from tortoise import fields, models
from tortoise.contrib import test
from tortoise.exceptions import OperationalError, IntegrityError

from tests import testmodels_postgres as testmodels
from tortoise.contrib.postgres.fields import VectorField # Make sure this import is correct

from typing import List, Iterable
import random


# ================== TESTCASE ============================
from types import ModuleType
from tortoise import Tortoise, connections

class VectorTestCase(test.SimpleTestCase):
    tortoise_test_modules: Iterable[str | ModuleType] = []

    async def _setUpDB(self) -> None:
        await super()._setUpDB()
        config = test.getDBConfig(app_label="models", modules=self.tortoise_test_modules or _MODULES)
        await Tortoise.init(config=config, _create_db=True)

        connection = connections.get("models")
        connection_alias = list(config['connections'].keys())[0] # Gets 'models'
        db_dialect = config['connections'][connection_alias]['engine']

        if "asyncpg" in db_dialect or "psycopg" in db_dialect:
            await connection.execute_script("CREATE EXTENSION IF NOT EXISTS vector;")
            await Tortoise.generate_schemas(safe=False)
        else:
            test.SkipTest("Not postgres.")

    async def _tearDownDB(self) -> None:
        await Tortoise._drop_databases()


# ================== VECTOR ============================

def create_random_vector(dimensions: int) -> List[float]:
    if dimensions <= 0:
        return []
    return [random.uniform(-1, 1) for _ in range(dimensions)]

@test.requireCapability(dialect="postgres")
class TestVectorField(VectorTestCase):
    """Test cases for the VectorField using a clean, isolated PostgreSQL+pgvector database.

    There's something weird with extensions and testing. Worked around it using
    inheritance...

    pgvector's real are 6-place; AlmostEqual assertions accordingly.

    """
    tortoise_test_modules = ["tests.testmodels_postgres"]

    async def test_vector_creation_and_retrieval(self):
        embedding_3_data = create_random_vector(3)
        embedding_1536_data = create_random_vector(1536)

        obj1 = await testmodels.VectorFields.create(
            embedding_3=embedding_3_data,
            embedding_1536=embedding_1536_data
        )
        self.assertIsNotNone(obj1.id)

        obj2 = await testmodels.VectorFields.get(id=obj1.id)

        self.assertEqual(obj1.id, obj2.id)

        self.assertEqual(len(obj2.embedding_3), 3)

        for original, retrieved in zip(embedding_3_data, obj2.embedding_3):
            self.assertAlmostEqual(original, retrieved, places=6)

        self.assertEqual(len(obj2.embedding_1536), 1536)

        for original, retrieved in zip(embedding_1536_data, obj2.embedding_1536):
            self.assertAlmostEqual(original, retrieved, places=6)

        self.assertIsInstance(obj2.embedding_1536, list)
        self.assertIsInstance(obj2.embedding_3, list)

    async def test_vector_invalid_dimensions(self):
        embedding_3_data = create_random_vector(4)

        with self.assertRaises(ValueError):
            await testmodels.VectorFields.create(embedding_3=embedding_3_data)

        embedding_1536_data = create_random_vector(1535)

        with self.assertRaises(ValueError):
            await testmodels.VectorFields.create(embedding_1536=embedding_1536_data)

    async def test_vector_empty_value(self):
        embedding_data = None

        obj1 = await testmodels.VectorFields.create(embedding_3=embedding_data)
        obj2 = await testmodels.VectorFields.get(id=obj1.id)
        self.assertIsNone(obj1.embedding_3)
        self.assertEqual(obj1.embedding_3, obj2.embedding_3)

    async def test_vector_search_types(self):
        vector1 = [0.000001, 0.000002, 0.000003]
        vector2 = [0.000123, 0.000456, 0.000789]
        vector3 = [1.234567, 2.345678, 3.456789]
        vector4 = [1.0, 2.0, 3.0]
        vector5 = [5.678901, 4.567890, 3.456789]

        obj1 = await testmodels.VectorFields.create(embedding_3=vector1)
        obj2 = await testmodels.VectorFields.create(embedding_3=vector2)
        obj3 = await testmodels.VectorFields.create(embedding_3=vector3)
        obj4 = await testmodels.VectorFields.create(embedding_3=vector4)
        obj5 = await testmodels.VectorFields.create(embedding_3=vector5)

        results = await testmodels.VectorFields.all().order_by_similarity(
            query_vector=vector1,
            search_type='L2_DISTANCE',
            vector_field_name="embedding_3"
        ).limit(3)

        order = [1, 2, 4]
        results_order = [i.id for i in results]
        self.assertListEqual(order, results_order)

        results = await testmodels.VectorFields.all().order_by_similarity(
            query_vector=vector1,
            search_type='INNER_PRODUCT',
            vector_field_name="embedding_3"
        ).limit(3)

        order = [5, 3, 4]
        results_order = [i.id for i in results]
        self.assertListEqual(order, results_order)

        results = await testmodels.VectorFields.all().order_by_similarity(
            query_vector=vector1,
            search_type='COSINE_SIMILARITY',
            vector_field_name="embedding_3"
        ).limit(3)

        order = [1, 4, 3]
        results_order = [i.id for i in results]
        self.assertListEqual(order, results_order)
