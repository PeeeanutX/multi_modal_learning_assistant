import unittest
import os
from src.ingestion.pdf_loader import extract_text_from_pdf


class TestPdfLoader(unittest.TestCase):

    def setUp(self):
        # Get the directory of the current script
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the data directory
        self.data_dir = os.path.join(self.test_dir, 'data')

    def test_extract_text_from_valid_pdf(self):
        pdf_path = 'tests/data/testojb.pdf'
        text = extract_text_from_pdf(pdf_path)
        self.assertIsNotNone(text)
        self.assertIn('Sample Text', text)

    def test_extract_text_from_empty_pdf(self):
        pdf_path = 'tests/data/testojb.pdf'
        text = extract_text_from_pdf(pdf_path)
        self.assertEqual(text, '')

    def test_extract_text_from_nonexistent_pdf(self):
        pdf_path = 'tests/data/testojb.pdf'
        text = extract_text_from_pdf(pdf_path)
        self.assertIsNone(text)

    def test_extract_text_from_encrypted_pdf(self):
        pdf_path = 'src/tests/data/testojb.pdf'
        text = extract_text_from_pdf(pdf_path)
        self.assertIsNone(text)


if __name__ == '__main__':
    unittest.main()