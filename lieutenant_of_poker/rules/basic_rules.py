"""
Basic heuristic rules for poker mistake detection.

These rules implement fundamental poker strategy guidelines that don't
require a solver - just basic hand reading and math.
"""

from typing import Optional, TYPE_CHECKING

from ..rules_engine import Rule, RuleContext, RuleViolation, Severity
from ..action_detector import PlayerAction
from ..card_detector import Card, Rank, Suit
from ..game_state import Street

if TYPE_CHECKING:
    from ..rules_engine import RulesEngine


# Premium hands - pairs and high cards that shouldn't be limped
PREMIUM_PAIRS = {Rank.ACE, Rank.KING, Rank.QUEEN, Rank.JACK, Rank.TEN}
PREMIUM_HIGH_CARDS = {
    (Rank.ACE, Rank.KING),
    (Rank.ACE, Rank.QUEEN),
    (Rank.ACE, Rank.JACK),
    (Rank.KING, Rank.QUEEN),
}


def is_pair(cards: list[Card]) -> bool:
    """Check if two cards form a pair."""
    if len(cards) != 2:
        return False
    return cards[0].rank == cards[1].rank


def is_suited(cards: list[Card]) -> bool:
    """Check if two cards are suited."""
    if len(cards) != 2:
        return False
    return cards[0].suit == cards[1].suit


def get_high_cards(cards: list[Card]) -> tuple[Rank, Rank]:
    """Get ranks sorted high to low."""
    if len(cards) != 2:
        return (Rank.TWO, Rank.TWO)
    ranks = sorted([c.rank for c in cards], key=lambda r: rank_value(r), reverse=True)
    return (ranks[0], ranks[1])


def rank_value(rank: Rank) -> int:
    """Get numeric value of a rank for comparison."""
    values = {
        Rank.TWO: 2,
        Rank.THREE: 3,
        Rank.FOUR: 4,
        Rank.FIVE: 5,
        Rank.SIX: 6,
        Rank.SEVEN: 7,
        Rank.EIGHT: 8,
        Rank.NINE: 9,
        Rank.TEN: 10,
        Rank.JACK: 11,
        Rank.QUEEN: 12,
        Rank.KING: 13,
        Rank.ACE: 14,
    }
    return values.get(rank, 0)


def is_premium_hand(cards: list[Card]) -> bool:
    """Check if hand is a premium starting hand."""
    if len(cards) != 2:
        return False

    # Premium pairs
    if is_pair(cards) and cards[0].rank in PREMIUM_PAIRS:
        return True

    # Premium high card combos
    high_cards = get_high_cards(cards)
    return high_cards in PREMIUM_HIGH_CARDS


def format_cards(cards: list[Card]) -> str:
    """Format cards for display."""
    return " ".join(str(c) for c in cards)


class FoldWhenCanCheckRule(Rule):
    """Warn if folding when check is available (no bet to call)."""

    name = "fold_when_can_check"
    description = "Warns when folding instead of checking for free"
    severity = Severity.ERROR

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # Only evaluate when hero is about to fold
        if context.action_about_to_take != PlayerAction.FOLD:
            return None

        # Check if check is available (no bet to call)
        if PlayerAction.CHECK in context.available_actions:
            return self._create_violation(
                message="Folding when you can check for free",
                suggestion="Check instead - there's no bet to call",
                severity=Severity.ERROR,
            )

        return None


class OpenLimpRule(Rule):
    """Warn about open-limping (calling the big blind as first to act)."""

    name = "open_limp"
    description = "Warns about limping when first to act preflop"
    severity = Severity.WARNING

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # Only applies preflop
        if not context.is_preflop:
            return None

        # Only check if hero is about to call
        if context.action_about_to_take != PlayerAction.CALL:
            return None

        # Check if this would be an open limp (first voluntary action)
        # An open limp is calling the BB when no one has raised
        hand = context.current_hand
        preflop_actions = [a for a in hand.actions if a.street == Street.PREFLOP]

        # If there are only blind posts or no raises, this is an open limp
        has_raise = any(
            a.action in (PlayerAction.RAISE, PlayerAction.BET) for a in preflop_actions
        )

        if not has_raise:
            cards_str = format_cards(context.hero_cards) if context.hero_cards else "?"

            return self._create_violation(
                message=f"Open-limping with {cards_str}",
                suggestion="Raise or fold - limping shows weakness and builds a smaller pot",
                cards=cards_str,
            )

        return None


class MinRaiseRule(Rule):
    """Warn about minimum raises."""

    name = "min_raise"
    description = "Warns about making minimum raises"
    severity = Severity.INFO

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # Only check if hero is about to raise
        if context.action_about_to_take != PlayerAction.RAISE:
            return None

        # We would need the raise amount to properly check this
        # For now, just note that min-raises are often suboptimal
        # This rule would be more useful with bet sizing detection

        return None  # Disabled until we have bet sizing


class PremiumHandLimpRule(Rule):
    """Warn about limping with premium hands."""

    name = "premium_limp"
    description = "Warns about limping with strong starting hands"
    severity = Severity.WARNING

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # Only applies preflop
        if not context.is_preflop:
            return None

        # Only check if hero is about to call (limp)
        if context.action_about_to_take != PlayerAction.CALL:
            return None

        # Need hero cards
        if len(context.hero_cards) != 2:
            return None

        # Check if this is a premium hand
        if is_premium_hand(context.hero_cards):
            cards_str = format_cards(context.hero_cards)

            return self._create_violation(
                message=f"Limping with premium hand {cards_str}",
                suggestion="Raise for value - premium hands should build the pot",
                cards=cards_str,
            )

        return None


class ShortStackRule(Rule):
    """Suggest push/fold strategy when short-stacked."""

    name = "short_stack"
    description = "Suggests push/fold when stack is below 10 big blinds"
    severity = Severity.INFO

    # Big blind threshold for push/fold strategy
    BB_THRESHOLD = 10

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # Only applies preflop
        if not context.is_preflop:
            return None

        # Check stack size in BBs
        hand = context.current_hand
        stack_bb = hand.hero_stack_bb

        if stack_bb <= 0:
            return None  # No stack info

        if stack_bb > self.BB_THRESHOLD:
            return None  # Not short-stacked

        # Hero is short-stacked
        # Check if they're trying to do something other than push/fold
        action = context.action_about_to_take

        if action == PlayerAction.CALL:
            return self._create_violation(
                message=f"Calling with only {stack_bb:.1f} BB stack",
                suggestion="With a short stack, consider all-in or fold - calling ties up chips",
                severity=Severity.WARNING,
                stack_bb=stack_bb,
            )

        if action == PlayerAction.RAISE:
            # Small raise instead of shove
            return self._create_violation(
                message=f"Raising with only {stack_bb:.1f} BB stack",
                suggestion="Consider shoving all-in for maximum fold equity",
                severity=Severity.INFO,
                stack_bb=stack_bb,
            )

        return None


class OversizedBetRule(Rule):
    """Warn about extremely oversized bets relative to pot."""

    name = "overbet"
    description = "Warns about bets significantly larger than the pot"
    severity = Severity.INFO

    # Threshold for what constitutes an overbet (as multiple of pot)
    OVERBET_THRESHOLD = 2.0

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # This rule would need bet sizing detection to work properly
        # Placeholder for future implementation
        return None


class CheckRaiseBluffRule(Rule):
    """Inform about check-raise situations."""

    name = "check_raise"
    description = "Notes check-raise opportunities and execution"
    severity = Severity.INFO

    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        # This is more of an informational rule
        # Would need position tracking and board texture analysis
        return None


def register_all(engine: "RulesEngine") -> None:
    """
    Register all basic rules with the engine.

    Args:
        engine: The rules engine to register with.
    """
    rules = [
        FoldWhenCanCheckRule(),
        OpenLimpRule(),
        MinRaiseRule(),
        PremiumHandLimpRule(),
        ShortStackRule(),
        OversizedBetRule(),
        CheckRaiseBluffRule(),
    ]

    for rule in rules:
        engine.register_rule(rule)
