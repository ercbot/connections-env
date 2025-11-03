# Changelog

## [0.1.1] - Adding PuzzGrid Rules

### Features

- Customizable ruleset configuration system to generate game prompts and scoring
- Added PuzzGrid.com ruleset
  - 3 max mistakes (vs NYT's 4)
  - Mistakes only count when 2 categories remain
  - No "one away" hints
  - Themes hidden until end
  - Theme guessing bonus round with intelligent matching

### Enhancements

- Track all guess history via state variables
- Auto-completion of final category (trivial guess)
- Simplified input validation reward

### Fixed

- Invalid guesses no longer count as mistakes
- Enhanced error messages with remaining words display
- Edge cases in guess parsing logic

## [0.1.0] - Initial Release

### Features

- Basic game implementation following NYT rules
  - 4 categories with 4 words each
  - 4 max mistakes
  - Mistakes always count from start
  - "One away" hints when 3/4 words correct
  - Immediate theme revelation
- XML-based parser for word guesses
- Basic scoring rubric
- HuggingFace dataset integration (`ericbotti/connections-puzzles`)
- State tracking (mistakes, found categories, found words)
- Input validation (word count, valid words, duplicate checking)
