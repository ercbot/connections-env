from verifiers import Rubric

from .rulesets import RulesetConfig
from .theme_matching import is_theme_match


class ConnectionsRubric(Rubric):
    def __init__(self, parser, ruleset_config: RulesetConfig, **kwargs):
        self.ruleset_config = ruleset_config
        super().__init__(parser=parser, **kwargs)
        # Word Grouping Reward Funcs
        # Validity Reward Funcs
        self.add_reward_func(self.guessed_correct_number_of_words, 0.25)
        self.add_reward_func(self.proportional_valid_words, 0.25)
        self.add_reward_func(self.all_words_valid, 0.25)
        # Correctness Reward Funcs
        self.add_reward_func(self.almost_found_categories, 0.5)
        self.add_reward_func(self.found_categories, 4.0)
        self.add_reward_func(self.efficiency_bonus)
        if not self.ruleset_config.end_game_theme_guessing:
            return
        # Theme Guessing Reward Funcs
        self.add_reward_func(self.attempted_theme_guessing, 0.25)
        self.add_reward_func(self.guessed_correct_number_of_themes, 0.5)
        self.add_reward_func(self.found_themes, 4.0)
        self.add_reward_func(self.found_all_themes_bonus, 1.0)

    def guessed_correct_number_of_words(self, completion, answer, state, info) -> float:
        """
        Proportion of guesses that had the correct number of words.
        """
        total_guesses = sum(1 for msg in completion if msg.get("role") == "assistant")
        if total_guesses == 0:
            return 0.0

        # Get expected group size from the first category
        expected_group_size = (
            len(info["categories"][0]["members"]) if info["categories"] else 4
        )

        valid_format_guesses = 0
        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if len(response) == expected_group_size:
                    valid_format_guesses += 1

        return valid_format_guesses / total_guesses

    def proportional_valid_words(self, completion, answer, state, info) -> float:
        """
        Average proportion of valid words across all guesses.
        """
        # Get all words in the game
        all_game_words = set()
        for category in info["categories"]:
            all_game_words.update([word.lower() for word in category["members"]])

        total_proportion = 0.0
        total_guesses = 0

        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if response:
                    valid_count = sum(1 for word in response if word in all_game_words)
                    proportion = valid_count / len(response)
                    total_proportion += proportion
                    total_guesses += 1

        return total_proportion / total_guesses if total_guesses > 0 else 0.0

    def all_words_valid(self, completion, answer, state, info) -> float:
        """
        Proportion of guesses where all words were valid.
        """
        # Get all words in the game
        all_game_words = set()
        for category in info["categories"]:
            all_game_words.update([word.lower() for word in category["members"]])

        # Get expected group size
        expected_group_size = (
            len(info["categories"][0]["members"]) if info["categories"] else 4
        )

        perfect_guesses = 0
        total_guesses = 0

        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if response and len(response) == expected_group_size:
                    valid_count = sum(1 for word in response if word in all_game_words)
                    if valid_count == len(response):
                        perfect_guesses += 1
                    total_guesses += 1

        return perfect_guesses / total_guesses if total_guesses > 0 else 0.0

    def almost_found_categories(self, completion, answer, state, info) -> float:
        """
        Count all guesses where the model guessed almost all words in a category (e.g., 3/4).
        Can only be given once per category, and is not given if that category was later correctly guessed.
        """
        # Get categories that were successfully found (don't reward for these)
        found_categories = state.get("found_categories", 0)

        # Track which categories we've already given "almost found" credit for
        almost_found_categories = set()

        # Get expected group size and calculate "almost" threshold
        expected_group_size = (
            len(info["categories"][0]["members"]) if info["categories"] else 4
        )
        almost_threshold = expected_group_size - 1  # e.g., 3 for 4-word groups

        # Go through all assistant messages
        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if (
                    len(response) == expected_group_size
                ):  # Only count proper-sized guesses
                    response_set = set(response)

                    # Check each category to see if this guess was almost correct
                    for i, category in enumerate(info["categories"]):
                        category_words = set(
                            [word.lower() for word in category["members"]]
                        )
                        overlap = len(response_set & category_words)

                        # If almost all words match and we haven't counted this category yet
                        if (
                            overlap == almost_threshold
                            and i not in almost_found_categories
                        ):
                            almost_found_categories.add(i)

        # Don't count categories that were later successfully found
        # This assumes categories are found in order, but we could make this more sophisticated
        categories_to_reward = max(0, len(almost_found_categories) - found_categories)

        return float(categories_to_reward)

    def found_categories(self, completion, answer, state, info) -> float:
        """
        Returns the proportion of categories found throughout the game (0.0-1.0).
        Normalized by total categories to ensure fair scoring across different puzzle sizes.
        """
        total_categories = len(info["categories"])
        if total_categories == 0:
            return 0.0
        return float(state.get("found_categories", 0)) / total_categories

    def efficiency_bonus(self, completion, answer, state) -> float:
        """
        Reward efficient play (fewer guesses per category found).
        Perfect efficiency (4 categories in 4 guesses) = 1.0
        """
        categories = state.get("found_categories", 0)
        if categories == 0:
            return 0.0

        # Count assistant messages (actual guesses)
        guesses = sum(1 for msg in completion if msg.get("role") == "assistant")
        if guesses == 0:
            return 0.0

        # Perfect efficiency is 1 guess per category
        efficiency = categories / guesses
        return min(efficiency, 1.0)  # Cap at 1.0 for perfect efficiency

    def attempted_theme_guessing(self, completion, answer, state, info) -> float:
        """
        Returns 1.0 if the AI attempted to guess any themes, 0.0 otherwise.
        Small reward to encourage theme guessing participation.
        """
        theme_guesses = state.get("theme_guesses", {})
        # Check if any theme guess was made (non-empty dict)
        return 1.0 if theme_guesses else 0.0

    def guessed_correct_number_of_themes(
        self, completion, answer, state, info
    ) -> float:
        """
        Returns 1.0 if the correct number of themes were guessed, 0.0 otherwise.
        """
        theme_guesses = state.get("theme_guesses", {})
        if not theme_guesses:
            return 0.0

        total_categories = len(info["categories"])
        num_guesses = len(theme_guesses)

        return 1.0 if num_guesses == total_categories else 0.0

    def found_themes(self, completion, answer, state, info) -> float:
        """
        Returns the proportion of themes correctly guessed (0.0-1.0).
        Awards points based on # themes found / total themes.
        """
        theme_guesses = state.get("theme_guesses", {})
        if not theme_guesses:
            return 0.0

        total_categories = len(info["categories"])
        if total_categories == 0:
            return 0.0

        # Count correct theme guesses
        correct_themes = 0
        for i, category in enumerate(info["categories"], 1):
            guessed_theme = theme_guesses.get(i, "")
            if guessed_theme:
                # Check if the guess matches using the same logic as the environment
                actual_theme = category["group"]
                linking_terms = category.get("linking_terms", [])

                # Use shared theme matching logic
                if is_theme_match(actual_theme, guessed_theme, linking_terms):
                    correct_themes += 1

        return correct_themes / total_categories

    def found_all_themes_bonus(self, completion, answer, state, info) -> float:
        """
        Returns 1.0 if all themes were correctly guessed, 0.0 otherwise.
        Bonus reward for finding all themes.
        """
        theme_guesses = state.get("theme_guesses", {})
        if not theme_guesses:
            return 0.0

        total_categories = len(info["categories"])
        if total_categories == 0:
            return 0.0

        # Check if all themes are correct
        for i, category in enumerate(info["categories"], 1):
            guessed_theme = theme_guesses.get(i, "")
            if not guessed_theme:
                return 0.0

            actual_theme = category["group"]
            linking_terms = category.get("linking_terms", [])

            # If any theme doesn't match, no bonus
            if not is_theme_match(actual_theme, guessed_theme, linking_terms):
                return 0.0

        # All themes match!
        return 1.0
