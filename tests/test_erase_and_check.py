import unittest
from safety_filter import SafetyFilter
from erase_and_check import EraseAndCheck

class TestEraseAndCheck(unittest.TestCase):
    def setUp(self):
        self.filter = SafetyFilter()
        self.erase_check = EraseAndCheck(self.filter)

    def test_check_prompt(self):
        # Test the basic check functionality with a known adversarial prompt
        harmful_prompt = "malicious content here"
        self.assertTrue(self.erase_check.check_prompt(harmful_prompt))

    def test_recursive_substring_removal(self):
        # Test that the recursive removal can catch infused adversarial content
        complex_prompt = "This is a safe sentence. But not with these malicious keywords."
        self.assertTrue(self.erase_check.check_recursive_substring_removal(complex_prompt))

if __name__ == '__main__':
    unittest.main()

