import unittest
from unittest.mock import patch, MagicMock
from rag_webbased import *  # Import from rag_webbased.py


class TestRAGWebBased(unittest.TestCase):

    @patch("rag_webbased.retriever.invoke")
    def test_retrieve_docs(self, mock_retriever):
        mock_retriever.return_value = [
            MagicMock(page_content="Page 1 content"),
            MagicMock(page_content="Page 2 content"),
        ]
        query = "Sample query"
        result = retrieve_docs(query)
        expected_result = "Page 1 content\nPage 2 content"
        self.assertEqual(result, expected_result)
        mock_retriever.assert_called_once_with(query)

    def test_stream(self):
        inputs = "Hello world"
        result = list(stream(inputs))
        expected_result = ["Hello ", "world "]
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
