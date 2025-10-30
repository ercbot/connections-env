from verifiers import Rubric

from .rulesets import RulesetConfig
from .theme_matching import is_theme_match


class ConnectionsRubric(Rubric):
    def __init__(self, parser, ruleset_config: RulesetConfig, **kwargs):
        self.ruleset_config = ruleset_config
        super().__init__(parser=parser, **kwargs)
        # Word Grouping Reward Funcs
        # Validity Reward Funcs
        self.add_reward_func(self.valid_guesses, 0.5)
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

    def valid_guesses(self, completion, answer, state, info) -> float:
        """
        Proportion of guesses that were valid (not invalid status).
        Valid statuses: correct, incorrect, one_away
        Invalid status: invalid
        """
        guess_history = state.get("guess_history", [])
        if not guess_history:
            return 0.0

        valid_guesses = sum(
            1 for guess in guess_history if guess["status"] != "invalid"
        )
        return valid_guesses / len(guess_history)

    def almost_found_categories(self, completion, answer, state, info) -> float:
        """
        Count "one away" guesses for categories that were never correctly found.
        This avoids double-counting: if you get one_away then later get it correct,
        you only get credit for the correct guess, not both.
        """
        guess_history = state.get("guess_history", [])

        # Track which categories were found correctly
        correctly_found_categories = set()
        # Track which categories had one_away guesses
        one_away_categories = set()

        for guess in guess_history:
            if guess["status"] == "correct" and guess.get("category_idx") is not None:
                correctly_found_categories.add(guess["category_idx"])
            elif (
                guess["status"] == "one_away" and guess.get("category_idx") is not None
            ):
                one_away_categories.add(guess["category_idx"])

        # Only reward one_away for categories that were never correctly found
        reward_categories = one_away_categories - correctly_found_categories

        return float(len(reward_categories))

    def found_categories(self, completion, answer, state, info) -> float:
        """
        Returns the proportion of categories found throughout the game (0.0-1.0).
        Normalized by total categories to ensure fair scoring across different puzzle sizes.
        """
        total_categories = len(info["categories"])
        if total_categories == 0:
            return 0.0
        return float(state.get("found_categories", 0)) / total_categories

    def efficiency_bonus(self, completion, answer, state, info) -> float:
        """
        Reward efficient play (fewer guesses to find all categories).
        Perfect efficiency = finding all categories in the minimum possible guesses.

        For a 4-category puzzle:
        - Minimum guesses needed: 3 (4th is auto-completed)
        - Perfect score: 3 guesses / 3 minimum = 1.0 (100%)
        - Good score: 4 guesses / 3 minimum = 0.75 (75%)
        - OK score: 6 guesses / 3 minimum = 0.5 (50%)
        """
        categories = state.get("found_categories", 0)
        if categories == 0:
            return 0.0

        # Count word guesses from guess_history (excludes theme guessing)
        guess_history = state.get("guess_history", [])
        if not guess_history:
            return 0.0

        # Calculate minimum guesses needed (total categories - 1, since last is auto-completed)
        total_categories = len(info["categories"])
        min_guesses_needed = total_categories - 1

        if min_guesses_needed == 0:
            return 1.0  # Edge case: if only 1 category total

        # Efficiency: minimum needed / actual guesses taken
        # Fewer guesses = higher efficiency
        efficiency = min_guesses_needed / len(guess_history)
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
