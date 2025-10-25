import logging
from typing import Tuple

from datasets import Dataset, load_dataset
from verifiers import MultiTurnEnv
from verifiers.types import Messages, State

from .dataset import prep_dataset
from .parser import ConnectionsParser
from .prompts import generate_system_prompt
from .rubric import ConnectionsRubric
from .rulesets import get_ruleset_config

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
        max_turns: int = 20,
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
        self, messages: Messages, state: State
    ) -> bool:
        """
        Check if the game is completed.
        Game ends when:
        1. All categories are found (win) and theme guessing is complete (if enabled)
        2. Max mistakes are made (lose)
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
        found_categories = state.get("found_categories", 0)
        # And check that we don't need to guess the theme
        is_theme_guessing_enabled = self.ruleset_config.end_game_theme_guessing

        if found_categories >= total_categories and not is_theme_guessing_enabled:
            # Without theme guessing - game is complete
            return True

        # Check if the the theme has been guessed
        theme_guesses = state.get("theme_guesses", {})

        if is_theme_guessing_enabled and theme_guesses:
            # Check if state is NOT an empty dict
            return True

        return False

    def _should_count_mistakes(self, state: State) -> bool:
        """
        Determine if mistakes should be counted based on ruleset configuration.
        """
        threshold = self.ruleset_config.mistakes_count_when_x_categories_remain
        if threshold == "any":
            return True

        remaining_categories = len(
            state.get("info", {}).get("categories", [])
        ) - state.get("found_categories", 0)
        return remaining_categories <= threshold

    def _is_theme_match(
        self, actual_theme: str, guessed_theme: str, linking_terms: list[str]
    ) -> bool:
        """
        Check if a guessed theme matches the actual theme using linking terms for flexibility.

        Args:
            actual_theme: The correct theme from the dataset
            guessed_theme: The user's guess
            linking_terms: List of linking words/phrases that should also count as correct

        Returns:
            True if the guess matches the theme or any linking terms
        """
        if not guessed_theme or guessed_theme.lower().strip() == "no guess":
            return False

        # Clean up strings for comparison
        actual_clean = actual_theme.lower().strip()
        guessed_clean = guessed_theme.lower().strip()

        # Exact match
        if actual_clean == guessed_clean:
            return True

        # Check if guess matches any linking terms
        for term in linking_terms:
            term_clean = term.lower().strip()

            # Exact match with linking term
            if guessed_clean == term_clean:
                return True

            # Check if guessed theme contains the linking term (or vice versa)
            if term_clean in guessed_clean or guessed_clean in term_clean:
                return True

        # Check if any word from the guess appears in the actual theme or linking terms
        guessed_words = set(guessed_clean.split())
        actual_words = set(actual_clean.split())

        # Check for word overlap with actual theme
        if guessed_words & actual_words:
            return True

        # Check for word overlap with linking terms
        for term in linking_terms:
            term_words = set(term.lower().split())
            if guessed_words & term_words:
                return True

        return False

    async def env_response(
        self, messages: Messages, state: State
    ) -> Tuple[Messages, State]:
        """
        Game logic, updates state and returns environment response to the AI based on the rules.

        1. Update game state based players last message
        2. Generate response based on new state
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
        if last_message["role"] != "assistant":
            return [], state

        # ============================================================
        # PART 1: UPDATE STATE
        # ============================================================
        state = await self._update_state(last_message["content"], state)

        # ============================================================
        # PART 2: GENERATE RESPONSE
        # ============================================================
        total_categories = len(state["info"]["categories"])
        mistakes = state.get("mistakes", 0)
        found_categories = state.get("found_categories", 0)

        # Still in word guessing phase
        if (
            mistakes < self.ruleset_config.max_mistakes
            and found_categories < total_categories
        ):
            response = self._generate_word_phase_response(state)
            return [{"role": "user", "content": response}], state

        # Transition to theme guessing phase
        elif (
            found_categories >= total_categories
            and self.ruleset_config.end_game_theme_guessing
            and "theme_guessing_started" not in state
        ):
            # Initialize theme guessing
            state["theme_guessing_started"] = True
            state["theme_guesses"] = {}

            # Create dynamic XML parser for theme guessing
            theme_fields = [
                f"category_{i}_guess" for i in range(1, total_categories + 1)
            ]

            from verifiers import XMLParser

            state["theme_parser"] = XMLParser(
                fields=theme_fields, answer_field=theme_fields[0]
            )

            # Present all found categories without themes
            categories_display = []
            xml_format_display = []
            for i, category in enumerate(state["info"]["categories"], 1):
                members_str = ", ".join(category["members"])
                categories_display.append(f"Category {i}: [{members_str}]")
                xml_format_display.append(
                    f"<category_{i}_guess>your theme guess</category_{i}_guess>"
                )

            categories_text = "\n".join(categories_display)
            xml_format_text = "\n".join(xml_format_display)

            return [
                {
                    "role": "user",
                    "content": f"🎉 Word guessing complete! Now guess the themes for bonus points:\n\n{categories_text}\n\nPlease guess the theme for each category using this format:\n\n{xml_format_text}",
                }
            ], state

        # In theme guessing phase (already started)
        elif "theme_guessing_started" in state and state.get("theme_guesses"):
            response = self._generate_theme_results_response(state)
            return [{"role": "user", "content": response}], state

        # Game over - no more responses needed (return empty messages)
        else:
            return [], state

    async def _update_state(self, last_message_content: str, state: State) -> State:
        """
        Update state based on player's last message. All logic inline.
        """
        total_categories = len(state["info"]["categories"])

        # Determine if we're in word phase or theme phase
        currently_in_word_phase = (
            state.get("mistakes", 0) < self.ruleset_config.max_mistakes
            and state.get("found_categories", 0) < total_categories
        )
        currently_in_theme_phase = "theme_guessing_started" in state

        # ============================================================
        # WORD GUESSING PHASE
        # ============================================================
        if currently_in_word_phase:
            # Parse the AI's guess
            try:
                guessed_words = self.parser.parse_answer_as_list(last_message_content)
            except Exception as e:
                # If parsing fails, count as a mistake
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1
                state["last_error"] = f"Error when trying to parse your guess: {str(e)}"
                return state

            # Get expected group size from the first category
            expected_group_size = (
                len(state["info"]["categories"][0]["members"])
                if state["info"]["categories"]
                else 4
            )

            # Validate the guess - check count
            if len(guessed_words) != expected_group_size:
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1
                state["last_error"] = f"Please guess exactly {expected_group_size} words. You guessed {len(guessed_words)}."
                return state

            # Check if all guessed words are valid
            invalid_words = [
                word for word in guessed_words if word not in state["all_words"]
            ]
            if invalid_words:
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1
                state["last_error"] = f"Invalid words: {', '.join(invalid_words)}. These words are not in the game."
                return state

            # Check if words are already found
            already_found = [word for word in guessed_words if word in state["found_words"]]
            if already_found:
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1
                state["last_error"] = f"Words already found: {', '.join(already_found)}."
                return state

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
                state["last_correct_category"] = correct_category
                state["last_error"] = None
            else:
                # Incorrect guess
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1

                # Check for "One Away" if ruleset allows
                state["last_one_away"] = False
                if self.ruleset_config.show_one_away_hints:
                    max_correct = 0
                    for category in state["info"]["categories"]:
                        category_words = set([word.lower() for word in category["members"]])
                        correct_count = len(set(guessed_words) & category_words)
                        if correct_count > max_correct:
                            max_correct = correct_count

                    if max_correct == expected_group_size - 1:
                        state["last_one_away"] = True

                state["last_correct_category"] = None
                state["last_error"] = None

        # ============================================================
        # THEME GUESSING PHASE
        # ============================================================
        elif currently_in_theme_phase:
            # Parse theme guesses using the dynamic XML parser
            try:
                theme_parser = state.get("theme_parser")
                if not theme_parser:
                    logger.error("Theme parser not found in state")
                    state["theme_parse_error"] = True
                    return state

                # Parse the XML content
                parsed = theme_parser.parse(last_message_content)
                guesses = {}

                # Extract all category guesses from the parsed namespace
                for i in range(1, total_categories + 1):
                    field_name = f"category_{i}_guess"
                    theme_guess = getattr(parsed, field_name, None)
                    if theme_guess and theme_guess.strip():
                        guesses[i] = theme_guess.strip()

                if not guesses:
                    state["theme_parse_error"] = True
                    return state

                # Store guesses in state for scoring
                state["theme_guesses"] = guesses
                state["theme_parse_error"] = False

            except Exception:
                state["theme_parse_error"] = True

        return state

    def _generate_word_phase_response(self, state: State) -> str:
        """
        Generate response text for word guessing phase based on current state.
        """
        total_categories = len(state["info"]["categories"])

        # Format mistake display if needed
        mistake_display = (
            f" Mistakes: {state['mistakes']}/{self.ruleset_config.max_mistakes}"
            if self._should_count_mistakes(state)
            else ""
        )

        # If there was an error in the last guess
        if state.get("last_error"):
            return f"{state['last_error']}{mistake_display}"

        # If the last guess was correct
        if state.get("last_correct_category"):
            correct_category = state["last_correct_category"]

            if self.ruleset_config.reveal_themes_immediately:
                response = f"Correct! Category {state['found_categories']}/{total_categories} found: {correct_category['group']} - {', '.join(correct_category['members'])}"
            else:
                response = f"Correct! Category {state['found_categories']}/{total_categories} found. Theme will be revealed at the end."

            # Add remaining words if not all categories found
            if state["found_categories"] < total_categories:
                remaining_words = [
                    word
                    for word in state["all_words"]
                    if word not in state["found_words"]
                ]
                response += f"\nRemaining words: {', '.join(remaining_words)}"

            return response

        # If the last guess was incorrect
        one_away_msg = "One away! " if state.get("last_one_away") else ""
        return f"{one_away_msg}Incorrect guess, try again.{mistake_display}"

    def _generate_theme_results_response(self, state: State) -> str:
        """
        Generate response text for theme guessing results.
        """
        # If there was a parsing error, show format instructions
        if state.get("theme_parse_error"):
            total_categories = len(state["info"]["categories"])
            xml_format_display = []
            for i in range(1, total_categories + 1):
                xml_format_display.append(
                    f"<category_{i}_guess>your theme guess</category_{i}_guess>"
                )
            xml_format_text = "\n".join(xml_format_display)
            return f"Please format your theme guesses using the XML format:\n\n{xml_format_text}"

        # Calculate theme guess accuracy for display
        correct_themes = 0
        total_categories = len(state["info"]["categories"])
        guesses = state.get("theme_guesses", {})

        theme_results = []
        for i, category in enumerate(state["info"]["categories"], 1):
            actual_theme = category["group"]
            guessed_theme = guesses.get(i, "No guess")

            # Enhanced theme matching with linking terms
            is_correct = self._is_theme_match(
                actual_theme, guessed_theme, category.get("linking_terms", [])
            )
            if is_correct:
                correct_themes += 1

            status = "✓" if is_correct else "✗"
            theme_results.append(
                f"Category {i}: {actual_theme} {status} (You guessed: {guessed_theme})"
            )

        results_text = "\n".join(theme_results)
        return f"Theme guessing complete!\n\n{results_text}\n\nYou got {correct_themes}/{total_categories} themes correct! Game finished."
