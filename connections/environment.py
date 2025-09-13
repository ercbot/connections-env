from typing import Any
from verifiers import MultiTurnEnv
from datasets import Dataset

from .dataset import prep_dataset
from .parser import ConnectionsParser
from .rubric import ConnectionsRubric
from .rulesets import get_ruleset_config, generate_system_prompt

from datasets import load_dataset


class ConnectionsEnv(MultiTurnEnv):
    """
    Environment for the Connections game.
    """

    def __init__(
        self,
        ruleset: str = "nyt",
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        max_turns: int = 10,
        **kwargs,
    ):
        # Get ruleset configuration
        self.ruleset_config = get_ruleset_config(ruleset)
        
        parser: ConnectionsParser = ConnectionsParser()
        rubric = ConnectionsRubric(parser=parser, ruleset_config=self.ruleset_config)

        # If no datasets provided, load and prep the default HuggingFace dataset
        if dataset is None:
            dataset_dict = load_dataset("ericbotti/connections-puzzles")
            dataset, eval_dataset = prep_dataset(dataset_dict)
        # Otherwise, assume user-provided datasets are already in correct format

        # Generate dynamic system prompt based on ruleset and expected group size
        # We'll use the first example to determine group size for now
        expected_group_size = 4  # Default fallback
        total_categories = 4     # Default fallback
        if dataset is not None and len(dataset) > 0:
            first_example = dataset[0]
            if "info" in first_example and "categories" in first_example["info"]:
                categories = first_example["info"]["categories"]
                if categories:
                    expected_group_size = len(categories[0]["members"])
                    total_categories = len(categories)

        system_prompt = generate_system_prompt(
            self.ruleset_config, 
            expected_group_size, 
            total_categories
        )

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

    async def is_completed(
        self, messages: list[dict], state: dict, **kwargs: Any
    ) -> bool:
        """
        Check if the game is completed.
        Game ends when:
        1. All categories are found (win)
        2. 4 mistakes are made (lose)
        3. Max turns reached
        """
        # Check if we've reached max turns
        if (
            len([msg for msg in messages if msg.get("role") == "assistant"])
            >= self.max_turns
        ):
            return True

        # Check if we've made 4 mistakes
        if state.get("mistakes", 0) >= 4:
            return True

        # Check if we've found all categories
        total_categories = len(state.get("info", {}).get("categories", []))
        if state.get("found_categories", 0) >= total_categories:
            return True

        return False

    async def env_response(
        self, messages: list[dict], state: dict, **kwargs: Any
    ) -> tuple[list[dict], dict]:
        """
        Generate environment response based on the AI's guess.
        """
        # Initialize state if not present
        if "mistakes" not in state:
            state["mistakes"] = 0
        if "found_categories" not in state:
            state["found_categories"] = 0
        if "found_words" not in state:
            state["found_words"] = set()
        if "all_words" not in state:
            # Extract all words from the categories structure
            all_words = []
            for category in state["info"]["categories"]:
                all_words.extend([word.lower() for word in category["members"]])
            state["all_words"] = set(all_words)

        # Get the AI's last response
        last_message = messages[-1]
        if last_message.get("role") != "assistant":
            return [
                {"role": "user", "content": "Please make a guess of 4 words."}
            ], state

        # Parse the AI's guess
        try:
            guessed_words = self.parser.parse_answer_as_list(last_message["content"])
        except Exception as e:
            print(e)
            print(last_message["content"])
            # If parsing fails, count as a mistake
            state["mistakes"] += 1
            return [
                {
                    "role": "user",
                    "content": f"Invalid format. Please guess exactly 4 words separated by commas. Mistakes: {state['mistakes']}/4",
                }
            ], state

        # Get expected group size from the first category (assume all same size)
        expected_group_size = (
            len(state["info"]["categories"][0]["members"])
            if state["info"]["categories"]
            else 4
        )

        # Validate the guess
        if len(guessed_words) != expected_group_size:
            state["mistakes"] += 1
            return [
                {
                    "role": "user",
                    "content": f"Please guess exactly {expected_group_size} words. You guessed {len(guessed_words)}. Mistakes: {state['mistakes']}/4",
                }
            ], state

        # Check if all guessed words are valid
        invalid_words = [
            word for word in guessed_words if word not in state["all_words"]
        ]
        if invalid_words:
            state["mistakes"] += 1
            return [
                {
                    "role": "user",
                    "content": f"Invalid words: {', '.join(invalid_words)}. These words are not in the game. Mistakes: {state['mistakes']}/4",
                }
            ], state

        # Check if words are already found
        already_found = [word for word in guessed_words if word in state["found_words"]]
        if already_found:
            state["mistakes"] += 1
            return [
                {
                    "role": "user",
                    "content": f"Words already found: {', '.join(already_found)}. Mistakes: {state['mistakes']}/4",
                }
            ], state

        # Check if the guess matches a category
        correct_category = None
        for category in state["info"]["categories"]:
            category_words = set([word.lower() for word in category["members"]])
            if set(guessed_words) == category_words:
                correct_category = category
                break

        if correct_category:
            # Correct guess!
            state["found_categories"] += 1
            state["found_words"].update(guessed_words)

            total_categories = len(state["info"]["categories"])
            response = f"Correct! Category {state['found_categories']}/{total_categories} found: {correct_category['group']} - {', '.join(correct_category['members'])}"

            if state["found_categories"] >= total_categories:
                response += "\n🎉 Congratulations! You've found all categories!"
            else:
                remaining_words = [
                    word
                    for word in state["all_words"]
                    if word not in state["found_words"]
                ]
                response += f"\nRemaining words: {', '.join(remaining_words)}"

        else:
            # Incorrect guess
            state["mistakes"] += 1

            # Check for "One Away" (3 correct words)
            max_correct = 0
            for category in state["info"]["categories"]:
                category_words = set([word.lower() for word in category["members"]])
                correct_count = len(set(guessed_words) & category_words)
                if correct_count > max_correct:
                    max_correct = correct_count

            if max_correct == 3:
                response = f"One away! Mistakes: {state['mistakes']}/4"
            else:
                response = f"Incorrect guess. Mistakes: {state['mistakes']}/4"

        return [{"role": "user", "content": response}], state
