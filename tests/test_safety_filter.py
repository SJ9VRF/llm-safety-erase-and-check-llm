import unittest
from safety_filter import SafetyFilter

class TestSafetyFilter(unittest.TestCase):
    def setUp(self):
        self.filter = SafetyFilter()

    def test_is_harmful_positive(self):
        # Test case where the prompt is known to be harmful
        harmful_prompt = "example harmful text"
        self.assertTrue(self.filter.is_harmful(harmful_prompt))

    def test_is_harmful_negative(self):
        # Test case where the prompt is known to be not harmful
        non_harmful_prompt = "example safe text"
        self.assertFalse(self.filter.is_harmful(non_harmful_prompt))

    def test_detect_complex_patterns(self):
        # Test detection of complex patterns
        prompt_with_homoglyphs = "somе malicious tеxt with cyrillic е's"
        self.assertTrue(self.filter.detect_complex_patterns(prompt_with_homoglyphs))

if __name__ == '__main__':
    unittest.main()

