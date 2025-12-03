"""
Validators for poker game states.

is_complete_frame: Validates single frame completeness (no missing data).
validate_transition: Validates state transitions according to Texas Hold'em rules.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from .game_state import GameState, Street, PlayerState
from .card_detector import Card
from .table_regions import HERO, seat_name


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
    CHIPS_DECREASED_WITHOUT_POT_INCREASE = auto()
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


def _get_street_index(street: Street) -> int:
    """Get the ordinal index of a street for comparison."""
    try:
        return STREET_ORDER.index(street)
    except ValueError:
        return 0  # Unknown streets treated as earliest


def _cards_to_set(cards: List[Card]) -> Set[Tuple[str, str]]:
    """Convert a list of cards to a set of (rank, suit) tuples for comparison."""
    return {(card.rank.name, card.suit.name) for card in cards}


def _card_to_str(card: Card) -> str:
    """Convert a card to a string representation."""
    return f"{card.rank.name}{card.suit.name[0]}"


def _cards_to_str(cards: List[Card]) -> str:
    """Convert a list of cards to a string representation."""
    return ", ".join(_card_to_str(c) for c in cards) if cards else "none"


def is_complete_frame(state: GameState) -> bool:
    """Check if a frame has all required values (no None in critical fields).

    This validates data completeness, not rule validity.
    Empty seats (chips=None) are allowed - only active players are checked.
    """
    if state.pot is None or state.hero_chips is None:
        return False
    if len(state.hero_cards) < 2:
        return False
    # Check that at least one non-hero player has chips (not all empty seats)
    has_active_player = False
    for player in state.players.values():
        if player.position == HERO:
            continue
        if player.chips is not None:
            has_active_player = True
            break
    return has_active_player


def _is_new_hand(prev: GameState, curr: GameState) -> bool:
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
    prev_idx = _get_street_index(prev.street)
    curr_idx = _get_street_index(curr.street)
    if prev_idx >= _get_street_index(Street.RIVER) and curr_idx <= _get_street_index(Street.PREFLOP):
        return True

    # Hero cards completely different AND pot significantly decreased
    # Both conditions required together to avoid false positives from OCR errors
    if len(prev.hero_cards) == 2 and len(curr.hero_cards) == 2:
        prev_set = _cards_to_set(prev.hero_cards)
        curr_set = _cards_to_set(curr.hero_cards)
        hero_cards_changed = not prev_set.intersection(curr_set)

        # Check if pot reset (decreased significantly or to small blinds value)
        pot_reset = False
        if prev.pot is not None and curr.pot is not None:
            # Pot decreased to less than 20% of previous (likely blinds)
            pot_reset = curr.pot < prev.pot * 0.2

        if hero_cards_changed and pot_reset:
            return True

    return False


def _validate_community_cards(prev: GameState, curr: GameState) -> List[Violation]:
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
        prev_cards = _cards_to_set(prev.community_cards[:prev_count])
        curr_cards_subset = _cards_to_set(curr.community_cards[:prev_count])
        if prev_cards != curr_cards_subset:
            violations.append(Violation(
                violation_type=ViolationType.COMMUNITY_CARDS_CHANGED,
                message=f"Existing community cards changed",
                previous_value=_cards_to_str(prev.community_cards),
                current_value=_cards_to_str(curr.community_cards[:prev_count]),
            ))

    return violations


def _validate_hero_cards(prev: GameState, curr: GameState) -> List[Violation]:
    """Validate hero cards don't change during a hand."""
    violations = []

    # Hero cards disappeared (had 2 cards, now have fewer)
    if len(prev.hero_cards) == 2 and len(curr.hero_cards) < 2:
        violations.append(Violation(
            violation_type=ViolationType.HERO_CARDS_DISAPPEARED,
            message=f"Hero cards disappeared (had {_cards_to_str(prev.hero_cards)})",
            previous_value=_cards_to_str(prev.hero_cards),
            current_value=_cards_to_str(curr.hero_cards),
        ))
    # If both states have hero cards, they should match
    elif len(prev.hero_cards) == 2 and len(curr.hero_cards) == 2:
        prev_set = _cards_to_set(prev.hero_cards)
        curr_set = _cards_to_set(curr.hero_cards)
        if prev_set != curr_set:
            violations.append(Violation(
                violation_type=ViolationType.HERO_CARDS_CHANGED,
                message="Hero cards changed mid-hand",
                previous_value=_cards_to_str(prev.hero_cards),
                current_value=_cards_to_str(curr.hero_cards),
            ))

    return violations


def _validate_no_duplicates(state: GameState) -> List[Violation]:
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
            duplicates.append(_card_to_str(card))
        seen.add(card_tuple)

    if duplicates:
        violations.append(Violation(
            violation_type=ViolationType.DUPLICATE_CARDS,
            message=f"Duplicate cards detected: {', '.join(duplicates)}",
            current_value=", ".join(duplicates),
        ))

    return violations


def _validate_pot(prev: GameState, curr: GameState) -> List[Violation]:
    """Validate pot changes are matched by chip changes."""
    violations = []

    if curr.pot is None:
        violations.append(Violation(
            violation_type=ViolationType.POT_DISAPPEARED,
            message=f"Pot is None",
            previous_value=str(prev.pot),
            current_value="None",
        ))
        return violations

    if curr.pot == prev.pot:
        return violations

    # Calculate total chip decrease
    total_chip_decrease = 0
    for pos in curr.players:
        if pos in prev.players:
            prev_chips = prev.players[pos].chips
            curr_chips = curr.players[pos].chips
            if prev_chips is not None and curr_chips is not None and curr_chips < prev_chips:
                total_chip_decrease += prev_chips - curr_chips

    if curr.pot < prev.pot:
        violations.append(Violation(
            violation_type=ViolationType.POT_DECREASED,
            message=f"Pot decreased from {prev.pot} to {curr.pot}",
            previous_value=str(prev.pot),
            current_value=str(curr.pot),
        ))
    elif total_chip_decrease == 0:
        violations.append(Violation(
            violation_type=ViolationType.POT_DECREASED,
            message=f"Pot changed from {prev.pot} to {curr.pot} without chip decrease",
            previous_value=str(prev.pot),
            current_value=str(curr.pot),
        ))

    return violations


def _validate_street(prev: GameState, curr: GameState) -> List[Violation]:
    """Validate street can only progress forward."""
    violations = []

    prev_idx = _get_street_index(prev.street)
    curr_idx = _get_street_index(curr.street)

    # Street went backwards (not to UNKNOWN/PREFLOP which could indicate new hand)
    if curr_idx < prev_idx and curr.street not in (Street.UNKNOWN, Street.PREFLOP):
        violations.append(Violation(
            violation_type=ViolationType.STREET_REGRESSION,
            message=f"Street regressed from {prev.street.name} to {curr.street.name}",
            previous_value=prev.street.name,
            current_value=curr.street.name,
        ))

    return violations


def _validate_chip_changes(prev: GameState, curr: GameState) -> List[Violation]:
    """Validate chip changes are matched by pot changes."""
    violations = []

    # Calculate pot change
    pot_change = 0
    if prev.pot is not None and curr.pot is not None:
        pot_change = curr.pot - prev.pot

    # Collect all player chips (including hero via players dict)
    for pos in curr.players:
        if pos not in prev.players:
            continue
        prev_chips = prev.players[pos].chips
        curr_chips = curr.players[pos].chips
        if prev_chips is None or curr_chips is None:
            continue

        chip_change = curr_chips - prev_chips
        if chip_change > 0:
            # Chips increased - always invalid (we don't track wins)
            violations.append(Violation(
                violation_type=ViolationType.CHIPS_INCREASED_WITHOUT_WIN,
                message=f"Player {seat_name(pos)} chips increased by {chip_change}",
                previous_value=str(prev_chips),
                current_value=str(curr_chips),
            ))
        elif chip_change < 0:
            # Chips decreased - pot should increase
            chip_decrease = -chip_change
            if pot_change < chip_decrease * 0.5:
                violations.append(Violation(
                    violation_type=ViolationType.CHIPS_DECREASED_WITHOUT_POT_INCREASE,
                    message=f"Player {seat_name(pos)} chips decreased by {chip_decrease} but pot only increased by {pot_change}",
                    previous_value=str(prev_chips),
                    current_value=str(curr_chips),
                ))

    return violations


def _validate_total_chips(prev: GameState, curr: GameState) -> List[Violation]:
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


def validate_transition(
    prev: GameState,
    curr: GameState,
    allow_new_hand: bool = True,
    check_chip_increases: bool = False,
) -> ValidationResult:
    """
    Validate a state transition from prev to curr.

    Args:
        prev: The previous game state.
        curr: The current (proposed) game state.
        allow_new_hand: Allow state resets that indicate a new hand.
        check_chip_increases: Flag chip increases as violations (disabled by default).

    Returns:
        ValidationResult with is_valid and list of violations.
    """
    violations = []

    # Check if this might be a new hand
    if allow_new_hand and _is_new_hand(prev, curr):
        return ValidationResult(is_valid=True, violations=[])

    # Always check these
    violations.extend(_validate_community_cards(prev, curr))
    violations.extend(_validate_hero_cards(prev, curr))
    violations.extend(_validate_no_duplicates(curr))
    violations.extend(_validate_pot(prev, curr))
    violations.extend(_validate_street(prev, curr))

    # Optional checks
    if check_chip_increases:
        violations.extend(_validate_chip_changes(prev, curr))

    return ValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )
