import re

from verifiers import XMLParser


class ConnectionsParser(XMLParser):
    """Special implementation of the XMLParser that can parse the answer as a list of items."""

    def __init__(self, **kwargs):
        fields = ["guess"]
        answer_field = "guess"
        super().__init__(fields=fields, answer_field=answer_field, **kwargs)

    def parse_answer_as_list(self, completion: str) -> tuple[list[str], int]:
        """
        Parse the completion to extract a list of items from the last guess tag.

        When multiple <guess> tags are present, the last complete tag is used.
        This handles cases where an LLM mentions <guess> in explanatory text
        before providing the actual guess.

        Returns:
            tuple[list[str], int]: A tuple of (items, num_guess_tags_found)
        """
        # Find all complete guess tags with their content
        guess_matches = re.findall(
            r"<guess>(.*?)</guess>", completion, re.DOTALL | re.IGNORECASE
        )

        num_guess_tags = len(guess_matches)

        if num_guess_tags == 0:
            return [], 0

        # Take the content of the last complete guess tag
        response = guess_matches[-1].strip()

        # Handle both list format [WORD1, WORD2, ...] and comma-separated format
        if response.startswith("[") and response.endswith("]"):
            response = response[1:-1]  # Remove brackets

        # Extract backticked items (don't split commas inside backticks)
        # Match pattern: backticked content (anything between backticks)
        pattern = r"`([^`]+)`"
        matches = re.findall(pattern, response)

        # Preserve original case for items
        items = [match.strip() for match in matches]
        return [item for item in items if item], num_guess_tags
