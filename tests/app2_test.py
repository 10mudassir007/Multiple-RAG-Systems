import unittest
from unittest.mock import patch, MagicMock
from rag_streamlit import generate

class TestRAGStreamlit(unittest.TestCase):

    @patch("rag_streamlit.retriver.get_relevant_documents")
    @patch("rag_streamlit.rag_chain.stream")
    def test_generate(self, mock_rag_chain, mock_retriever):
        mock_retriever.return_value = [
            MagicMock(page_content="Document 1 content"),
            MagicMock(page_content="Document 2 content"),
        ]
        mock_rag_chain.return_value = ["This is a test response."]
        query = "Test query"
        result = generate(query)
        self.assertEqual(result, ["This is a test response."])
        mock_retriever.assert_called_once_with(query=query)
        mock_rag_chain.assert_called_once_with({
            "question": query,
            "documents": "Document 1 content\nDocument 2 content",
        })

if __name__ == "__main__":
    unittest.main()
