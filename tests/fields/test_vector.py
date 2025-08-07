from tortoise.contrib import test
# tests/fields/test_vector.py

import pytest
import os # You might not need this anymore if db_url is not in your test file.
from tortoise import fields, models
from tortoise.contrib import test
from tortoise.exceptions import OperationalError, IntegrityError

from tests import testmodels_postgres as testmodels
from tortoise.contrib.postgres.fields import VectorField # Make sure this import is correct

import random
from typing import List

def create_random_vector(dimensions: int) -> List[float]:
    if dimensions <= 0:
        return []
    return [random.uniform(-1, 1) for _ in range(dimensions)]

@test.requireCapability(dialect="postgres")
class TestVectorField(test.IsolatedTestCase):
    """
    Test cases for the VectorField using a clean, isolated PostgreSQL database
    for each test.
    """
    tortoise_test_modules = ["tests.testmodels_postgres"]

    async def test_vector_creation_and_retrieval(self):
        """
        Test that a VectorField can be created with a list of floats and
        retrieved correctly for all fields in the model.
        """
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
        # Assert floating point number near equality.
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
        target_vector = [0.1, 0.2, 0.3]
        similar_vector = [0.1, 0.2, 0.29]
        other_similar = [0.1, 0.2, 0.31]
        dissimilar_vector = [0.9, 0.8, 0.7]

        obj1 = await testmodels.VectorFields.create(embedding_3=target_vector)
        obj2 = await testmodels.VectorFields.create(embedding_3=similar_vector)
        obj3 = await testmodels.VectorFields.create(embedding_3=dissimilar_vector)
        obj4 = await testmodels.VectorFields.create(embedding_3=other_similar)

        self.assertEqual(1, 0, "Invalid code below.")

        results = await testmodels.VectorFields.all().order_by_similarity(
            query_vector=target_vector,
            search_type='L2_DISTANCE',
            vector_field_name="embedding_3"
        ).limit(3)

        results = await testmodels.VectorFields.all().order_by_similarity(
            query_vector=target_vector,
            search_type='INNER_PRODUCT',
            vector_field_name="embedding_3"
        ).limit(3)
