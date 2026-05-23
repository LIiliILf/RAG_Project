import unittest

import numpy as np

from embeddings import cosine_similarity


class EmbeddingUtilityTests(unittest.TestCase):
    def test_cosine_similarity_returns_expected_shape(self):
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype="float32",
        )

        scores = cosine_similarity(query, documents)

        self.assertEqual(scores.shape, (3,))

    def test_cosine_similarity_ranks_same_direction_highest(self):
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype="float32",
        )

        scores = cosine_similarity(query, documents)
        best_index = int(np.argmax(scores))

        self.assertEqual(best_index, 1)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=5)

    def test_cosine_similarity_rejects_empty_documents(self):
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.empty((0, 2), dtype="float32")

        with self.assertRaises(ValueError):
            cosine_similarity(query, documents)


if __name__ == "__main__":
    unittest.main()
