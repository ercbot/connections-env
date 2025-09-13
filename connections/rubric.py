from verifiers import Rubric
from .rulesets import RulesetConfig


class ConnectionsRubric(Rubric):
    def __init__(self, parser, ruleset_config: RulesetConfig, **kwargs):
        self.ruleset_config = ruleset_config
        super().__init__(parser=parser, **kwargs)
        # Validity Reward Funcs
        # - Rewards for correctly following the rules of the game
        self.add_reward_func(self.guessed_4_words, 0.25)
        self.add_reward_func(self.proportional_valid_words, 0.25)
        self.add_reward_func(self.all_words_valid, 0.25)
        # Correctness Reward Funcs
        # - Rewards for playing the game well
        self.add_reward_func(self.almost_found_categories, 0.5)
        self.add_reward_func(self.found_categories)
        self.add_reward_func(self.efficiency_bonus)

    def guessed_4_words(self, completion, answer, state) -> float:
        """
        Proportion of guesses that had exactly 4 words.
        """
        total_guesses = sum(1 for msg in completion if msg.get("role") == "assistant")
        if total_guesses == 0:
            return 0.0

        valid_format_guesses = 0
        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if len(response) == 4:
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

        # Track which categories we've already given "almost found" credit for
        perfect_guesses = 0
        total_guesses = 0

        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if response and len(response) == 4:
                    valid_count = sum(1 for word in response if word in all_game_words)
                    if valid_count == len(response):
                        perfect_guesses += 1
                    total_guesses += 1

        return perfect_guesses / total_guesses if total_guesses > 0 else 0.0

    def almost_found_categories(self, completion, answer, state, info) -> float:
        """
        Count all guesses where the model guessed 3/4 words in a category.
        Can only be given once per category, and is not given if that category was later correctly guessed.
        """
        # Get categories that were successfully found (don't reward for these)
        found_categories = state.get("found_categories", 0)

        # Track which categories we've already given "almost found" credit for
        almost_found_categories = set()

        # Go through all assistant messages
        for msg in completion:
            if msg.get("role") == "assistant":
                response = self.parser.parse_answer_as_list(msg["content"])
                if len(response) == 4:  # Only count proper 4-word guesses
                    response_set = set(response)

                    # Check each category to see if this guess had 3/4 words from it
                    for i, category in enumerate(info["categories"]):
                        category_words = set(
                            [word.lower() for word in category["members"]]
                        )
                        overlap = len(response_set & category_words)

                        # If exactly 3/4 words match and we haven't counted this category yet
                        if overlap == 3 and i not in almost_found_categories:
                            almost_found_categories.add(i)

        # Don't count categories that were later successfully found
        # This assumes categories are found in order, but we could make this more sophisticated
        categories_to_reward = max(0, len(almost_found_categories) - found_categories)

        return float(categories_to_reward)

    def found_categories(self, completion, answer, state) -> float:
        """
        Returns the total number of categories found throughout the game.
        """
        return float(state.get("found_categories", 0))

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