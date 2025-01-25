import re

class SafetyFilter:
    def __init__(self, model_name="bert-base-uncased", model_path="path/to/custom_model"):
        """
        Initialize the SafetyFilter with a pre-trained or fine-tuned model that is specialized for detecting harmful content, potentially adversarial.

        Parameters:
            model_name (str): Name of the model to load, e.g., a BERT variant fine-tuned on adversarial data.
            model_path (str): Path to the custom-trained model directory.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def is_harmful(self, prompt):
        """
        Determine if a given prompt is harmful by using the loaded model.
        Incorporates additional checks for adversarial patterns and subtleties.

        Parameters:
            prompt (str): The text prompt to evaluate.

        Returns:
            bool: True if the prompt is harmful, False otherwise.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scores = torch.softmax(logits, dim=1)
        harmful_score = scores[:, 1]  # Assuming index '1' is harmful
        return harmful_score.item() > 0.5  # Threshold can be adjusted


    def check_adversarial_patterns(self, prompt):
        """
        Checks for specific adversarial patterns that might not be typical 'harmful' content
        but are tailored to exploit model weaknesses. This includes checking for:
        
        - Unnatural phrase repetitions
        - Embedding of trigger words or phrases
        - Syntax that is unusual or overly complex
        - Use of homoglyphs or visually deceptive characters

        Parameters:
            prompt (str): The text prompt to check.

        Returns:
            bool: True if adversarial patterns are detected, False otherwise.
        """
        # Check for unnatural repetitions
        if self.detect_repetitions(prompt):
            return True
        
        # Check for trigger words or phrases
        if self.detect_trigger_words(prompt):
            return True
        
        # Check for unusual or complex syntax
        if self.detect_unusual_syntax(prompt):
            return True
        
        # Check for homoglyphs or visually deceptive characters
        if self.detect_homoglyphs(prompt):
            return True

        return False

    def detect_repetitions(self, text):
        """
        Detects unnatural repetitions in the text which might be used to confuse the model.
        """
        pattern = r"\b(\w+)\s+\1\b"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return len(matches) > 2  # More than two repetitions could be suspicious

    def detect_trigger_words(self, text):
        """
        Detects the presence of known trigger words or phrases that are often used in adversarial attacks.
        """
        trigger_words = ['example_trigger_word1', 'example_trigger_word2']
        return any(word in text for word in trigger_words)

    def detect_unusual_syntax(self, text):
        """
        Checks for syntax that is unusually complex or structured in a way that might be trying to exploit model weaknesses.
        """
        # Hypothetical example: excessively long sentences or misuse of grammatical structures
        if len(text) > 200 and ',' not in text:
            return True  # Unusually long sentence without commas could be suspicious
        return False

    def detect_homoglyphs(self, text):
        """
        Detects the use of homoglyphs or visually deceptive characters that might be intended to mislead the model.
        """
        homoglyphs_pattern = r'[а-яА-Я]'  # Cyrillic characters that look like Latin characters
        return re.search(homoglyphs_pattern, text) is not None
