---
ruleset: puzzgrid
prompt_type: system
max_mistakes: 3
mistakes_count_when_x_categories_remain: 2
show_one_away_hints: False
reveal_themes_immediately: False
end_game_theme_guessing: True
token_count: 370
---

You are playing *Connections*. Items are grouped by theme. Find the group members.

## Round 1
Select items that belong together:
- All correct = group completed
- Early guesses don't count as mistakes
- When 2 or fewer categories remain, mistakes start counting
- 3 mistakes during counting phase = game over


Repeat until all groups found (win) or too many mistakes (lose).
Themes revealed only at game end.
Examples:
- Theme: FISH, Items: [`BASS`, `TROUT`, `SALMON`, `TUNA`]
- Theme: FIRE ____, Items: [`ANT`, `DRILL`, `ISLAND`, `OPAL`]
- Theme: QUICKLY, Items: [`IN A FLASH`, `AT ONCE`, `RIGHT AWAY`, `IN NO TIME`]

Tips:
- Each puzzle has one solution. Items may fit multiple categories - choose carefully.
- All groups have the same amount of items, specified at game start.


Note: The items are usually words, but can also be whole phrases, abbreviations, fragments of words, emojis, or other pieces of text. Whatever is between the backticks is the exact text of the item.

In your response, indicate your guess with this format:
<guess>[`ITEM1`, `ITEM2`, `ITEM3`, ...]</guess>

You can only make one guess per response, but your response can including reasoning or notes.


## Round 2
After finding all item groups, guess each category's theme for bonus points.

Format:
<category_1_guess>theme</category_1_guess>
<category_2_guess>theme</category_2_guess>
(etc.)

Themes can be: common category, words following/preceding a phrase, shared property, or other connection.

