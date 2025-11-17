import logging
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

from datasets import Dataset, load_dataset
from verifiers import MultiTurnEnv
from verifiers.types import Messages, State

from .dataset import prep_dataset
from .parser import ConnectionsParser
from .prompts import generate_system_prompt
from .rubric import ConnectionsRubric
from .rulesets import get_ruleset_config
from .utils import is_theme_match, words_to_string


@dataclass
class GuessRecord:
    """Records information about a single guess attempt.

    Status meanings:
    - invalid: Guess failed validation (wrong count, invalid words, or already-found words)
    - incorrect: Valid guess but doesn't match any category
    - one_away: Valid guess that's one word away from matching a category
    - correct: Valid guess that exactly matches a category
    """

    words: list[str]  # The words that were guessed
    status: Literal["invalid", "incorrect", "one_away", "correct"]
    category_idx: Optional[int] = (
        None  # Which category (for one_away or correct status)
    )


logger = logging.getLogger(__name__)


class ConnectionsEnv(MultiTurnEnv):
    """
    Environment for the Connections game.
    """

    def __init__(
        self,
        ruleset: str = "nyt",
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        system_prompt: str | None = None,
        are_datasets_raw: bool = True,
        max_turns: int = 10,
        **kwargs,
    ):
        # Get ruleset configuration
        self.ruleset_config = get_ruleset_config(ruleset)

        parser: ConnectionsParser = ConnectionsParser()
        rubric = ConnectionsRubric(parser=parser, ruleset_config=self.ruleset_config)

        if dataset is None:
            dataset = load_dataset("ericbotti/connections-puzzles", split="train_rl")
        if eval_dataset is None:
            eval_dataset = load_dataset("ericbotti/connections-puzzles", split="test")

        if are_datasets_raw:
            dataset = prep_dataset(dataset, self.ruleset_config)
            eval_dataset = prep_dataset(eval_dataset, self.ruleset_config)

        # Use provided system prompt or generate one using ruleset configuration
        system_prompt = system_prompt or generate_system_prompt(self.ruleset_config)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

    async def setup_state(self, state: State, **kwargs) -> State:
        """
        Setup state with optional resumption from a previous guess_history.

        If info contains "resumed_from_guess_history", we reconstruct the game state
        to match where it was when it was truncated, including:
        - mistakes count
        - found_categories count
        - remaining_words list
        - guess_history list

        This allows us to "resume" a game from a checkpoint for doctoring purposes.
        """
        info = state.get("info", {})
        resumed_history = info.get("resumed_from_guess_history")

        if resumed_history:
            # Reconstruct state from the resumed guess history
            categories = info.get("categories", [])
            all_words = info.get("all_words", [])

            # Initialize base state
            state["mistakes"] = 0
            state["found_categories"] = 0
            state["remaining_words"] = all_words.copy()  # Start with all words
            state["guess_history"] = []

            # Replay the guess history to reconstruct state
            for guess_record in resumed_history:
                status = guess_record.get("status")
                category_idx = guess_record.get("category_idx")

                # Convert category_idx to int if it's not None (JSON may load as float)
                if category_idx is not None:
                    category_idx = int(category_idx)

                # Add to guess history
                state["guess_history"].append(guess_record)

                if status == "correct":
                    # Correctly guessed category
                    state["found_categories"] += 1
                    if category_idx is not None and category_idx < len(categories):
                        found_category_words = categories[category_idx]["members"]
                        # Remove found words from remaining_words, preserving order
                        found_words_lower = {w.lower() for w in found_category_words}
                        state["remaining_words"] = [
                            w
                            for w in state["remaining_words"]
                            if w.lower() not in found_words_lower
                        ]
                elif status == "incorrect" or status == "one_away":
                    # Count mistakes based on ruleset configuration
                    # Use a temporary state dict to call _should_count_mistakes
                    temp_state = {
                        "info": info,
                        "found_categories": state["found_categories"],
                    }
                    if self._should_count_mistakes(temp_state):
                        state["mistakes"] += 1
                # invalid guesses don't count as mistakes

            logger.info(
                f"Resumed game state from guess_history: "
                f"{state['found_categories']} categories found, "
                f"{state['mistakes']} mistakes"
            )

        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
        """
        Check if the game is completed.
        Sets state["complete_reason"] to track why the game ended:
        - "max_turns_reached": Maximum turn limit reached
        - "prompt_too_long": Context became too long
        - "max_mistakes": Player made too many mistakes
        - "all_categories_found": All categories found (without theme guessing)
        - "theme_guessing_complete": Game complete including theme guessing
        """
        # Check parent class completion conditions (max turns, prompt too long)
        if await self.max_turns_reached(state):
            state["complete_reason"] = "max_turns_reached"
            return True

        if await self.prompt_too_long(state):
            state["complete_reason"] = "prompt_too_long"
            return True

        # Check if we've made max mistakes
        if state.get("mistakes", 0) >= self.ruleset_config.max_mistakes:
            state["complete_reason"] = "max_mistakes"
            return True

        # Check if we've found all categories
        total_categories = len(state.get("info", {}).get("categories", []))
        found_categories = state.get("found_categories", 0)
        # And check that we don't need to guess the theme
        is_theme_guessing_enabled = self.ruleset_config.end_game_theme_guessing

        if found_categories >= total_categories and not is_theme_guessing_enabled:
            # Without theme guessing - game is complete
            state["complete_reason"] = "all_categories_found"
            return True

        # Check if the the theme has been guessed
        theme_guesses = state.get("theme_guesses", {})

        if is_theme_guessing_enabled and theme_guesses:
            # Check if state is NOT an empty dict
            state["complete_reason"] = "theme_guessing_complete"
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
        if "remaining_words" not in state:
            # Get shuffled word order from info (stored during dataset prep)
            all_words = state.get("info", {}).get("all_words", [])
            state["remaining_words"] = all_words.copy()
        if "guess_history" not in state:
            # Track all guesses with metadata for reward calculation
            state["guess_history"] = []

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
                members_str = words_to_string(category["members"])
                categories_display.append(f"Category {i}: {members_str}")
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
                # If parsing fails, don't count as mistake - just ask to retry
                state["last_error"] = (
                    f"Invalid guess format: {str(e)}. This did not cost a mistake, try again."
                )
                # Record invalid guess
                guess_record = GuessRecord(words=[], status="invalid")
                state["guess_history"].append(asdict(guess_record))
                return state

            # Get expected group size from the first category
            expected_group_size = (
                len(state["info"]["categories"][0]["members"])
                if state["info"]["categories"]
                else 4
            )

            # Validate the guess - check count
            if len(guessed_words) != expected_group_size:
                remaining_words_str = words_to_string(state["remaining_words"])
                state["last_error"] = (
                    f"Invalid guess, you guessed {len(guessed_words)} words but need exactly {expected_group_size}. "
                    f"This did not cost a mistake, try again.\n"
                    f"Remaining words: {remaining_words_str}"
                )
                # Record invalid guess
                guess_record = GuessRecord(words=guessed_words, status="invalid")
                state["guess_history"].append(asdict(guess_record))
                return state

            # Check if all guessed words are valid (case-insensitive)
            # Check against all words from categories (not just remaining)
            all_words_from_categories = set()
            for category in state["info"]["categories"]:
                all_words_from_categories.update(category["members"])
            all_words_lower = {word.lower() for word in all_words_from_categories}
            invalid_words = [
                word for word in guessed_words if word.lower() not in all_words_lower
            ]
            if invalid_words:
                word_label = "word" if len(invalid_words) == 1 else "words"
                invalid_words_str = words_to_string(invalid_words)
                remaining_words_str = words_to_string(state["remaining_words"])
                state["last_error"] = (
                    f"Invalid guess, the {word_label} {invalid_words_str} {'is' if len(invalid_words) == 1 else 'are'} not in the game. "
                    f"This did not cost a mistake, try again.\n"
                    f"Remaining words: {remaining_words_str}"
                )
                # Record invalid guess
                guess_record = GuessRecord(words=guessed_words, status="invalid")
                state["guess_history"].append(asdict(guess_record))
                return state

            # Check if all words guessed are in remaining_words
            remaining_words_lower = {word.lower() for word in state["remaining_words"]}
            words_not_in_remaining_words = [
                word
                for word in guessed_words
                if word.lower() not in remaining_words_lower
            ]
            if words_not_in_remaining_words:
                state["last_error"] = (
                    f"Invalid guess, the words {words_to_string(words_not_in_remaining_words)} are have already been guessed. This did not cost a mistake, try again."
                )
                # Record invalid guess
                guess_record = GuessRecord(words=guessed_words, status="invalid")
                state["guess_history"].append(asdict(guess_record))
                return state

            # Check if the guess matches a category
            correct_category = None
            correct_category_idx = None
            for idx, category in enumerate(state["info"]["categories"]):
                category_words = set([word.lower() for word in category["members"]])
                if set(guessed_words) == category_words:
                    correct_category = category
                    correct_category_idx = idx
                    break

            if correct_category:
                # Correct guess! Store the original capitalized words from the category
                state["found_categories"] += 1
                found_category_words = correct_category["members"]
                # Remove found words from remaining_words, preserving order
                found_words_lower = {w.lower() for w in found_category_words}
                state["remaining_words"] = [
                    w
                    for w in state["remaining_words"]
                    if w.lower() not in found_words_lower
                ]
                state["last_correct_category"] = correct_category
                state["last_error"] = None
                # Record valid + correct guess
                guess_record = GuessRecord(
                    words=guessed_words,
                    status="correct",
                    category_idx=correct_category_idx,
                )
                state["guess_history"].append(asdict(guess_record))

                # Auto-complete if only 1 category remains
                if state["found_categories"] == total_categories - 1:
                    # Find the last remaining category
                    remaining_words_lower = {
                        word.lower() for word in state["remaining_words"]
                    }
                    for idx, category in enumerate(state["info"]["categories"]):
                        category_words_lower = {
                            word.lower() for word in category["members"]
                        }
                        # Check if this category hasn't been found yet
                        # A category is found if ALL its words are NOT in remaining_words
                        if (
                            category_words_lower.intersection(remaining_words_lower)
                            == category_words_lower
                        ):
                            # Auto-complete this category
                            state["found_categories"] += 1
                            auto_completed_words = category["members"]
                            # Remove auto-completed words from remaining_words, preserving order
                            auto_completed_words_lower = {
                                w.lower() for w in auto_completed_words
                            }
                            state["remaining_words"] = [
                                w
                                for w in state["remaining_words"]
                                if w.lower() not in auto_completed_words_lower
                            ]
                            state["last_auto_completed_category"] = category
                            # Record auto-completed category
                            auto_guess_record = GuessRecord(
                                words=category["members"],
                                status="correct",
                                category_idx=idx,
                            )
                            state["guess_history"].append(asdict(auto_guess_record))
                            break
            else:
                # Incorrect guess
                if self._should_count_mistakes(state):
                    state["mistakes"] += 1

                # Check for "One Away" if ruleset allows
                # IMPORTANT: Only count as one_away if it was a VALID guess
                # (already validated: correct count, all words in game, no already-found words)
                guess_status = "incorrect"
                one_away_category_idx = None

                if self.ruleset_config.show_one_away_hints:
                    max_correct = 0
                    best_category_idx = None
                    for idx, category in enumerate(state["info"]["categories"]):
                        category_words = set(
                            [word.lower() for word in category["members"]]
                        )
                        correct_count = len(set(guessed_words) & category_words)
                        if correct_count > max_correct:
                            max_correct = correct_count
                            best_category_idx = idx

                    if max_correct == expected_group_size - 1:
                        guess_status = "one_away"
                        one_away_category_idx = best_category_idx

                state["last_one_away"] = guess_status == "one_away"
                state["last_correct_category"] = None
                state["last_error"] = None
                # Record valid but incorrect guess
                guess_record = GuessRecord(
                    words=guessed_words,
                    status=guess_status,
                    category_idx=one_away_category_idx,
                )
                state["guess_history"].append(asdict(guess_record))

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
                members_str = words_to_string(correct_category["members"])
                response = f"Correct! Category {state['found_categories'] - (1 if state.get('last_auto_completed_category') else 0)}/{total_categories} found: {correct_category['group']} - {members_str}"
            else:
                response = f"Correct! Category {state['found_categories'] - (1 if state.get('last_auto_completed_category') else 0)}/{total_categories} found. Theme will be revealed at the end."

            # If last category was auto-completed, add that info
            if state.get("last_auto_completed_category"):
                auto_cat = state["last_auto_completed_category"]
                if self.ruleset_config.reveal_themes_immediately:
                    members_str = words_to_string(auto_cat["members"])
                    response += f"\n\nAll categories found! The last category was: {auto_cat['group']} - {members_str}"
                else:
                    response += "\n\nAll categories found! The last category has been auto-completed."
                # Clear the flag
                state.pop("last_auto_completed_category", None)

            # Add remaining words if not all categories found
            elif state["found_categories"] < total_categories:
                remaining_words_str = words_to_string(state["remaining_words"])
                response += f"\nRemaining words: {remaining_words_str}"

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
            guessed_theme = guesses.get(i, None)

            # Enhanced theme matching with linking terms
            is_correct = is_theme_match(
                actual_theme, guessed_theme, category.get("linking_terms", [])
            )
            if is_correct:
                correct_themes += 1

            status = "✓" if is_correct else "✗"
            guess_display = guessed_theme if guessed_theme else "(no guess)"
            theme_results.append(
                f"Category {i}: {actual_theme} {status} (You guessed: {guess_display})"
            )

        results_text = "\n".join(theme_results)
        return f"Theme guessing complete!\n\n{results_text}\n\nYou got {correct_themes}/{total_categories} themes correct! Game finished."
