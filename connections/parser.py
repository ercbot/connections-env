import re

from verifiers import XMLParser

connections_round_prompt_template = """\
The word grid currently looks like so:

{word_grid}

You have {number_of_mistakes} mistakes left.
"""


class ConnectionsParser(XMLParser):
    """Special implementation of the XMLParser that can parse the answer as a list of words."""

    def __init__(self, **kwargs):
        fields = ["guess"]
        answer_field = "guess"
        super().__init__(fields=fields, answer_field=answer_field, **kwargs)

    def parse_answer_as_list(self, completion: str) -> list[str]:
        # Check for multiple guess tags first
        guess_matches = re.findall(
            r"<guess>.*?</guess>", completion, re.DOTALL | re.IGNORECASE
        )
        if len(guess_matches) > 1:
            raise ValueError(
                f"Multiple guess tags found ({len(guess_matches)}). Please submit only one guess."
            )

        response = self.parse_answer(completion)
        if response is None:
            return []
        # Convert to string if it's a SimpleNamespace object
        if hasattr(response, "guess"):
            response = response.guess
        elif not isinstance(response, str):
            return []

        # Handle both list format [WORD1, WORD2, ...] and comma-separated format
        response = response.strip()
        if response.startswith("[") and response.endswith("]"):
            response = response[1:-1]  # Remove brackets

        # Extract backticked words (don't split commas inside backticks)
        # Match pattern: backticked content (anything between backticks)
        pattern = r"`([^`]+)`"
        matches = re.findall(pattern, response)

        words = [match.lower().strip() for match in matches]
        return [word for word in words if word]  # Filter out empty strings
