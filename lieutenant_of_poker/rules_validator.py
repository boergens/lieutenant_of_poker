"""
Texas Hold'em rules validator for state transitions.

Validates that game state transitions are legal according to Texas Hold'em rules.
Used to filter out frames with OCR errors or detection failures that would
result in impossible game states.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from .game_state import GameState, Street, PlayerState
from .card_detector import Card
from .table_regions import PlayerPosition


class ViolationType(Enum):
    """Types of rule violations."""
    COMMUNITY_CARDS_DECREASED = auto()
    COMMUNITY_CARDS_INVALID_COUNT = auto()
    COMMUNITY_CARDS_CHANGED = auto()
    HERO_CARDS_CHANGED = auto()
    HERO_CARDS_DISAPPEARED = auto()
    DUPLICATE_CARDS = auto()
    POT_DECREASED = auto()
    POT_DISAPPEARED = auto()
    STREET_REGRESSION = auto()
    CHIPS_INCREASED_WITHOUT_WIN = auto()
    CHIPS_DISAPPEARED = auto()
    TOTAL_CHIPS_CHANGED = auto()


@dataclass
class Violation:
    """A rule violation detected during state transition."""
    violation_type: ViolationType
    message: str
    previous_value: Optional[str] = None
    current_value: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validating a state transition."""
    is_valid: bool
    violations: List[Violation]

    @property
    def violation_types(self) -> Set[ViolationType]:
        """Get set of violation types."""
        return {v.violation_type for v in self.violations}


# Valid community card counts and their corresponding streets
VALID_COMMUNITY_COUNTS = {0, 3, 4, 5}

# Street order for progression checking
STREET_ORDER = [
    Street.UNKNOWN,
    Street.PREFLOP,
    Street.FLOP,
    Street.TURN,
    Street.RIVER,
    Street.SHOWDOWN,
]


def get_street_index(street: Street) -> int:
    """Get the ordinal index of a street for comparison."""
    try:
        return STREET_ORDER.index(street)
    except ValueError:
        return 0  # Unknown streets treated as earliest


def cards_to_set(cards: List[Card]) -> Set[Tuple[str, str]]:
    """Convert a list of cards to a set of (rank, suit) tuples for comparison."""
    return {(card.rank.name, card.suit.name) for card in cards}


def card_to_str(card: Card) -> str:
    """Convert a card to a string representation."""
    return f"{card.rank.name}{card.suit.name[0]}"


def cards_to_str(cards: List[Card]) -> str:
    """Convert a list of cards to a string representation."""
    return ", ".join(card_to_str(c) for c in cards) if cards else "none"


class RulesValidator:
    """
    Validates game state transitions according to Texas Hold'em rules.

    This validator checks that state changes are legal and filters out
    frames that would create impossible game states (likely due to OCR errors).
    """

    def __init__(
        self,
        check_community_cards: bool = True,
        check_hero_cards: bool = True,
        check_duplicates: bool = True,
        check_pot: bool = True,
        check_street: bool = True,
        check_chip_increases: bool = False,  # Disabled by default - wins are valid
        check_total_chips: bool = False,  # Disabled by default - hard to track accurately
        allow_new_hand: bool = True,  # Allow detecting new hand starts
    ):
        """
        Initialize the validator with configurable checks.

        Args:
            check_community_cards: Validate community card progression.
            check_hero_cards: Validate hero cards don't change mid-hand.
            check_duplicates: Check for duplicate cards across all positions.
            check_pot: Validate pot can only increase.
            check_street: Validate street can only progress forward.
            check_chip_increases: Flag chip increases as violations (disabled by default).
            check_total_chips: Validate total chips are conserved (disabled by default).
            allow_new_hand: Allow state resets that indicate a new hand.
        """
        self.check_community_cards = check_community_cards
        self.check_hero_cards = check_hero_cards
        self.check_duplicates = check_duplicates
        self.check_pot = check_pot
        self.check_street = check_street
        self.check_chip_increases = check_chip_increases
        self.check_total_chips = check_total_chips
        self.allow_new_hand = allow_new_hand

    def is_new_hand(self, prev: GameState, curr: GameState) -> bool:
        """
        Detect if the current state represents a new hand.

        A new hand is detected when multiple indicators align:
        - Community cards reset to 0 (strongest indicator)
        - Street regressed to PREFLOP from late street
        - Hero cards changed AND pot reset (both required together)

        Single indicators alone (like hero cards changing) are NOT enough
        to declare a new hand, as they could be OCR errors.
        """
        # Community cards went from some cards to none - strong indicator
        if len(prev.community_cards) > 0 and len(curr.community_cards) == 0:
            return True

        # Street went from SHOWDOWN/RIVER back to PREFLOP - strong indicator
        prev_idx = get_street_index(prev.street)
        curr_idx = get_street_index(curr.street)
        if prev_idx >= get_street_index(Street.RIVER) and curr_idx <= get_street_index(Street.PREFLOP):
            return True

        # Hero cards completely different AND pot significantly decreased
        # Both conditions required together to avoid false positives from OCR errors
        if len(prev.hero_cards) == 2 and len(curr.hero_cards) == 2:
            prev_set = cards_to_set(prev.hero_cards)
            curr_set = cards_to_set(curr.hero_cards)
            hero_cards_changed = not prev_set.intersection(curr_set)

            # Check if pot reset (decreased significantly or to small blinds value)
            pot_reset = False
            if prev.pot is not None and curr.pot is not None:
                # Pot decreased to less than 20% of previous (likely blinds)
                pot_reset = curr.pot < prev.pot * 0.2

            if hero_cards_changed and pot_reset:
                return True

        return False

    def validate_transition(
        self,
        prev: GameState,
        curr: GameState,
    ) -> ValidationResult:
        """
        Validate a state transition from prev to curr.

        Args:
            prev: The previous game state.
            curr: The current (proposed) game state.

        Returns:
            ValidationResult with is_valid and list of violations.
        """
        violations = []

        # Check if this might be a new hand
        if self.allow_new_hand and self.is_new_hand(prev, curr):
            # New hand - allow the transition
            return ValidationResult(is_valid=True, violations=[])

        # Check community cards
        if self.check_community_cards:
            violations.extend(self._validate_community_cards(prev, curr))

        # Check hero cards
        if self.check_hero_cards:
            violations.extend(self._validate_hero_cards(prev, curr))

        # Check for duplicate cards
        if self.check_duplicates:
            violations.extend(self._validate_no_duplicates(curr))

        # Check pot
        if self.check_pot:
            violations.extend(self._validate_pot(prev, curr))

        # Check street progression
        if self.check_street:
            violations.extend(self._validate_street(prev, curr))

        # Check chip increases (optional)
        if self.check_chip_increases:
            violations.extend(self._validate_chip_changes(prev, curr))

        # Check total chips conservation (optional)
        if self.check_total_chips:
            violations.extend(self._validate_total_chips(prev, curr))

        return ValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
        )

    def _validate_community_cards(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate community card transitions."""
        violations = []
        prev_count = len(prev.community_cards)
        curr_count = len(curr.community_cards)

        # Community cards can't decrease (except for new hand, handled above)
        if curr_count < prev_count:
            violations.append(Violation(
                violation_type=ViolationType.COMMUNITY_CARDS_DECREASED,
                message=f"Community cards decreased from {prev_count} to {curr_count}",
                previous_value=str(prev_count),
                current_value=str(curr_count),
            ))

        # Community cards must be valid count (0, 3, 4, or 5)
        if curr_count not in VALID_COMMUNITY_COUNTS:
            violations.append(Violation(
                violation_type=ViolationType.COMMUNITY_CARDS_INVALID_COUNT,
                message=f"Invalid community card count: {curr_count} (must be 0, 3, 4, or 5)",
                current_value=str(curr_count),
            ))

        # Existing community cards shouldn't change
        if curr_count >= prev_count > 0:
            prev_cards = cards_to_set(prev.community_cards[:prev_count])
            curr_cards_subset = cards_to_set(curr.community_cards[:prev_count])
            if prev_cards != curr_cards_subset:
                violations.append(Violation(
                    violation_type=ViolationType.COMMUNITY_CARDS_CHANGED,
                    message=f"Existing community cards changed",
                    previous_value=cards_to_str(prev.community_cards),
                    current_value=cards_to_str(curr.community_cards[:prev_count]),
                ))

        return violations

    def _validate_hero_cards(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate hero cards don't change during a hand."""
        violations = []

        # Hero cards disappeared (had 2 cards, now have fewer)
        if len(prev.hero_cards) == 2 and len(curr.hero_cards) < 2:
            violations.append(Violation(
                violation_type=ViolationType.HERO_CARDS_DISAPPEARED,
                message=f"Hero cards disappeared (had {cards_to_str(prev.hero_cards)})",
                previous_value=cards_to_str(prev.hero_cards),
                current_value=cards_to_str(curr.hero_cards),
            ))
        # If both states have hero cards, they should match
        elif len(prev.hero_cards) == 2 and len(curr.hero_cards) == 2:
            prev_set = cards_to_set(prev.hero_cards)
            curr_set = cards_to_set(curr.hero_cards)
            if prev_set != curr_set:
                violations.append(Violation(
                    violation_type=ViolationType.HERO_CARDS_CHANGED,
                    message="Hero cards changed mid-hand",
                    previous_value=cards_to_str(prev.hero_cards),
                    current_value=cards_to_str(curr.hero_cards),
                ))

        return violations

    def _validate_no_duplicates(self, state: GameState) -> List[Violation]:
        """Check for duplicate cards in the current state."""
        violations = []
        all_cards = []

        # Collect all visible cards
        all_cards.extend(state.hero_cards)
        all_cards.extend(state.community_cards)
        for player in state.players.values():
            all_cards.extend(player.cards)

        # Check for duplicates
        seen = set()
        duplicates = []
        for card in all_cards:
            card_tuple = (card.rank.name, card.suit.name)
            if card_tuple in seen:
                duplicates.append(card_to_str(card))
            seen.add(card_tuple)

        if duplicates:
            violations.append(Violation(
                violation_type=ViolationType.DUPLICATE_CARDS,
                message=f"Duplicate cards detected: {', '.join(duplicates)}",
                current_value=", ".join(duplicates),
            ))

        return violations

    def _validate_pot(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate pot can only increase during a hand."""
        violations = []

        # Pot disappeared (went from value to None) - likely OCR failure
        if prev.pot is not None and curr.pot is None:
            violations.append(Violation(
                violation_type=ViolationType.POT_DISAPPEARED,
                message=f"Pot disappeared (was {prev.pot})",
                previous_value=str(prev.pot),
                current_value="None",
            ))
        elif prev.pot is not None and curr.pot is not None:
            # Pot decreased significantly (small decreases might be rounding)
            # Allow pot to reset to 0 or small value for new hand
            if curr.pot < prev.pot:
                # Only flag if it's not a reset to blinds (indicating new hand)
                # Typical blinds are small, so we use a threshold
                if curr.pot > 0 and curr.pot > prev.pot * 0.1:
                    violations.append(Violation(
                        violation_type=ViolationType.POT_DECREASED,
                        message=f"Pot decreased from {prev.pot} to {curr.pot}",
                        previous_value=str(prev.pot),
                        current_value=str(curr.pot),
                    ))

        return violations

    def _validate_street(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate street can only progress forward."""
        violations = []

        prev_idx = get_street_index(prev.street)
        curr_idx = get_street_index(curr.street)

        # Street went backwards (not to UNKNOWN/PREFLOP which could indicate new hand)
        if curr_idx < prev_idx and curr.street not in (Street.UNKNOWN, Street.PREFLOP):
            violations.append(Violation(
                violation_type=ViolationType.STREET_REGRESSION,
                message=f"Street regressed from {prev.street.name} to {curr.street.name}",
                previous_value=prev.street.name,
                current_value=curr.street.name,
            ))

        return violations

    def _validate_chip_changes(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate chip increases are only from winning pots."""
        violations = []

        # Check hero chips
        if prev.hero_chips is not None and curr.hero_chips is not None:
            if curr.hero_chips > prev.hero_chips:
                # This could be a legitimate pot win, so we check if pot decreased
                pot_decreased = (
                    prev.pot is not None and
                    curr.pot is not None and
                    curr.pot < prev.pot
                )
                if not pot_decreased:
                    violations.append(Violation(
                        violation_type=ViolationType.CHIPS_INCREASED_WITHOUT_WIN,
                        message=f"Hero chips increased without pot decrease",
                        previous_value=str(prev.hero_chips),
                        current_value=str(curr.hero_chips),
                    ))

        # Check opponent chips
        for pos in curr.players:
            if pos in prev.players:
                prev_chips = prev.players[pos].chips
                curr_chips = curr.players[pos].chips
                if prev_chips is not None and curr_chips is not None:
                    if curr_chips > prev_chips:
                        pot_decreased = (
                            prev.pot is not None and
                            curr.pot is not None and
                            curr.pot < prev.pot
                        )
                        if not pot_decreased:
                            violations.append(Violation(
                                violation_type=ViolationType.CHIPS_INCREASED_WITHOUT_WIN,
                                message=f"Player {pos.name} chips increased without pot decrease",
                                previous_value=str(prev_chips),
                                current_value=str(curr_chips),
                            ))

        return violations

    def _validate_total_chips(
        self,
        prev: GameState,
        curr: GameState,
    ) -> List[Violation]:
        """Validate total chips in play are conserved."""
        violations = []

        def total_chips(state: GameState) -> Optional[int]:
            """Calculate total chips in the state."""
            total = 0
            if state.pot is not None:
                total += state.pot
            if state.hero_chips is not None:
                total += state.hero_chips
            for player in state.players.values():
                if player.chips is not None:
                    total += player.chips
            return total

        prev_total = total_chips(prev)
        curr_total = total_chips(curr)

        if prev_total is not None and curr_total is not None:
            # Allow some tolerance for OCR errors (5%)
            tolerance = prev_total * 0.05
            if abs(curr_total - prev_total) > tolerance:
                violations.append(Violation(
                    violation_type=ViolationType.TOTAL_CHIPS_CHANGED,
                    message=f"Total chips changed from {prev_total} to {curr_total}",
                    previous_value=str(prev_total),
                    current_value=str(curr_total),
                ))

        return violations


def filter_illegal_states(
    states: List[GameState],
    validator: Optional[RulesValidator] = None,
) -> List[GameState]:
    """
    Filter out states that would create illegal transitions.

    Args:
        states: List of game states to filter.
        validator: RulesValidator instance (uses defaults if None).

    Returns:
        List of valid game states with illegal transitions removed.
    """
    if not states:
        return []

    if validator is None:
        validator = RulesValidator()

    # First state is always included
    valid_states = [states[0]]

    for curr_state in states[1:]:
        prev_state = valid_states[-1]
        result = validator.validate_transition(prev_state, curr_state)

        if result.is_valid:
            valid_states.append(curr_state)
        # If invalid, skip this state (don't add to valid_states)

    return valid_states


def validate_state_sequence(
    states: List[GameState],
    validator: Optional[RulesValidator] = None,
) -> List[Tuple[int, ValidationResult]]:
    """
    Validate a sequence of states and return all violations.

    Args:
        states: List of game states to validate.
        validator: RulesValidator instance (uses defaults if None).

    Returns:
        List of (index, ValidationResult) for each invalid transition.
    """
    if len(states) < 2:
        return []

    if validator is None:
        validator = RulesValidator()

    invalid_transitions = []

    for i in range(1, len(states)):
        result = validator.validate_transition(states[i - 1], states[i])
        if not result.is_valid:
            invalid_transitions.append((i, result))

    return invalid_transitions
