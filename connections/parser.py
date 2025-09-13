from verifiers import XMLParser


connections_round_prompt_template = """\
The word grid currently looks like so:

{word_grid}

You have {number_of_mistakes} mistakes left.
"""


class ConnectionsParser(XMLParser):
    def __init__(self, **kwargs):
        fields = ["guess"]
        answer_field = "guess"
        super().__init__(fields=fields, answer_field=answer_field, **kwargs)

    def parse_answer_as_list(self, completion: str) -> list[str]:
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

        words = [
            word.lower().strip().strip('"').strip("'") for word in response.split(",")
        ]
        return [word for word in words if word]  # Filter out empty strings
