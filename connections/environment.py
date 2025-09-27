import logging
from typing import Any
from verifiers import MultiTurnEnv
from datasets import Dataset

from .dataset import prep_dataset
from .parser import ConnectionsParser
from .rubric import ConnectionsRubric
from .rulesets import get_ruleset_config
from .prompts import generate_system_prompt

from datasets import load_dataset

logger = logging.getLogger(__name__)


class ConnectionsEnv(MultiTurnEnv):
    """
    Environment for the Connections game.
    """

    def __init__(
        self,
        ruleset: str = "puzzgrid",
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
            dataset, eval_dataset = prep_dataset(dataset_dict, self.ruleset_config)
        # Otherwise, assume user-provided datasets are already in correct format

        # Generate system prompt based on ruleset configuration
        system_prompt = generate_system_prompt(self.ruleset_config)

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

        # Check if we've made max mistakes
        if state.get("mistakes", 0) >= self.ruleset_config.max_mistakes:
            return True

        # Check if we've found all categories
        total_categories = len(state.get("info", {}).get("categories", []))
        if state.get("found_categories", 0) >= total_categories:
            return True

        return False

    def _should_count_mistakes(self, state: dict) -> bool:
        """
        Determine if mistakes should be counted based on ruleset configuration.
        """
        remaining_categories = len(
            state.get("info", {}).get("categories", [])
        ) - state.get("found_categories", 0)
        return remaining_categories <= self.ruleset_config.mistakes_start_counting_at

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
            logger.error(f"Failed to parse guess: {e}")
            logger.error(f"Content: {last_message['content']}")
            # If parsing fails, count as a mistake (if mistakes are being counted)
            if self._should_count_mistakes(state):
                state["mistakes"] += 1
            mistake_display = (
                f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
                if self._should_count_mistakes(state)
                else ""
            )
            return [
                {
                    "role": "user",
                    "content": f"Invalid format. Please guess exactly 4 words separated by commas.{mistake_display}",
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
            if self._should_count_mistakes(state):
                state["mistakes"] += 1
            mistake_display = (
                f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
                if self._should_count_mistakes(state)
                else ""
            )
            return [
                {
                    "role": "user",
                    "content": f"Please guess exactly {expected_group_size} words. You guessed {len(guessed_words)}.{mistake_display}",
                }
            ], state

        # Check if all guessed words are valid
        invalid_words = [
            word for word in guessed_words if word not in state["all_words"]
        ]
        if invalid_words:
            if self._should_count_mistakes(state):
                state["mistakes"] += 1
            mistake_display = (
                f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
                if self._should_count_mistakes(state)
                else ""
            )
            return [
                {
                    "role": "user",
                    "content": f"Invalid words: {', '.join(invalid_words)}. These words are not in the game.{mistake_display}",
                }
            ], state

        # Check if words are already found
        already_found = [word for word in guessed_words if word in state["found_words"]]
        if already_found:
            if self._should_count_mistakes(state):
                state["mistakes"] += 1
            mistake_display = (
                f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
                if self._should_count_mistakes(state)
                else ""
            )
            return [
                {
                    "role": "user",
                    "content": f"Words already found: {', '.join(already_found)}.{mistake_display}",
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

            # Build response based on theme revelation setting
            if self.ruleset_config.reveal_themes_immediately:
                response = f"Correct! Category {state['found_categories']}/{total_categories} found: {correct_category['group']} - {', '.join(correct_category['members'])}"
            else:
                response = f"Correct! Category {state['found_categories']}/{total_categories} found. Theme will be revealed at the end."

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
            if self._should_count_mistakes(state):
                state["mistakes"] += 1

            # Check for "One Away" (3 correct words) if ruleset allows
            one_away_msg = ""
            if self.ruleset_config.show_one_away_hints:
                max_correct = 0
                for category in state["info"]["categories"]:
                    category_words = set([word.lower() for word in category["members"]])
                    correct_count = len(set(guessed_words) & category_words)
                    if correct_count > max_correct:
                        max_correct = correct_count

                if max_correct == expected_group_size - 1:  # One away (e.g., 3/4)
                    one_away_msg = "One away! "

            mistake_display = (
                f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
                if self._should_count_mistakes(state)
                else ""
            )
            response = f"{one_away_msg}Incorrect guess.{mistake_display}"

        return [{"role": "user", "content": response}], state
