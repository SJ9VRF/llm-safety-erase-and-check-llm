import re
import numpy as np

class EraseAndCheck:
    def __init__(self, safety_filter, max_token_length=20):
        """
        Initializes the EraseAndCheck object with a SafetyFilter instance and a maximum token length for erasing.

        Parameters:
            safety_filter (SafetyFilter): An instance of SafetyFilter to evaluate prompts.
            max_token_length (int): The maximum number of tokens to consider for erasing in the check process.
        """
        self.safety_filter = safety_filter
        self.max_token_length = max_token_length

    def check_prompt(self, prompt):
        """
        Primary method to check if a prompt is harmful by trying different modifications.

        Parameters:
            prompt (str): The text prompt to check.

        Returns:
            bool: True if any version of the prompt is detected as harmful, otherwise False.
        """
        # Check the original prompt
        if self.safety_filter.evaluate_prompt(prompt):
            return True

        # Check for adversarial patterns using advanced techniques
        if self.detect_complex_patterns(prompt):
            return True

        # Recursive check for infused adversarial content
        if self.check_recursive_substring_removal(prompt):
            return True

        return False

    def check_recursive_substring_removal(self, prompt, start_index=0):
        """
        Recursively removes tokens from the prompt to handle dispersed adversarial tokens.

        Parameters:
            prompt (str): The text prompt to evaluate.
            start_index (int): Current start index for recursive removal.

        Returns:
            bool: True if any recursively altered version of the prompt is harmful.
        """
        if self.safety_filter.evaluate_prompt(prompt):
            return True

        for i in range(start_index, len(prompt)):
            # Generate new prompt by skipping the current token
            new_prompt = prompt[:i] + prompt[i+1:]
            if self.check_recursive_substring_removal(new_prompt, i):
                return True
        return False

    def detect_complex_patterns(self, prompt):
        """
        Detects complex adversarial patterns such as homoglyphs, unusual syntax, and semantically unusual phrases.
    
        Parameters:
            prompt (str): The text prompt to check.
    
        Returns:
            bool: True if complex adversarial patterns are detected.
        """
        # Check for homoglyphs
        if re.search(r'[а-яА-Я]', prompt):  # Example of Cyrillic characters resembling Latin letters
            return True
    
        # Check for unusual syntax
        if re.search(r"(\b\w+\b\s*){10,}\b\w+\b[,.!?;:]", prompt):  # Unusually long sentences with minimal punctuation
            return True
    
        # Check for semantically unusual phrases
        # This is a placeholder for complexity; you might use a model to detect these based on a trained dataset
        unusual_phrases = ["colorless green ideas sleep furiously"]  # Example of a semantically nonsensical sentence
        if any(phrase in prompt for phrase in unusual_phrases):
            return True
    
        # Check for pattern disruption
        if re.search(r'\b\w+\b\s+\w+\b\s+\w+\b\s+fish\s+\w+\b\s+\w+\b\s+\w+\b', prompt):  # Unexpected word in a common pattern
            return True
    
        # Check for keyword stuffing
        if re.findall(r'\b(keyword)\b', prompt).count('keyword') > 3:  # Arbitrary number to detect stuffing
            return True
    
        return False


    def ml_based_adversarial_detection(self, prompt):
        """
        Uses a machine learning model trained to detect adversarial patterns within text that are typically missed by simpler methods.
    
        Parameters:
            prompt (str): The text prompt to evaluate.
    
        Returns:
            bool: True if the model detects adversarial content.
        """
        # Convert the prompt into a format suitable for the model (e.g., tokenization)
        input_features = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Run the model prediction
        with torch.no_grad():  # Ensure the model is in evaluation mode and not training mode
            logits = self.adv_model(input_features)
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get the predicted class (1 for adversarial, 0 for non-adversarial)
        predicted_class = probabilities.argmax(dim=-1).item()
    
        # Optionally, inspect the confidence of the prediction
        confidence = probabilities.max().item()
        print(f"Detection confidence: {confidence*100:.2f}%")
    
        # Return True if the predicted class is '1' (adversarial)
        return predicted_class == 1


    def check_adversarial_insertion(self, prompt):
        """
        Method to check for adversarial insertions by dynamically removing sections of the prompt.

        Parameters:
            prompt (str): The text prompt to evaluate.

        Returns:
            bool: True if any altered version of the prompt is harmful.
        """
        for start in range(len(prompt)):
            for end in range(start + 1, len(prompt) + 1):
                sub_prompt = prompt[:start] + prompt[end:]
                if self.safety_filter.evaluate_prompt(sub_prompt):
                    return True
        return False
