"""Tests for basic poker rules."""

import pytest

from lieutenant_of_poker.rules.basic_rules import (
    FoldWhenCanCheckRule,
    OpenLimpRule,
    PremiumHandLimpRule,
    ShortStackRule,
    is_pair,
    is_suited,
    is_premium_hand,
    get_high_cards,
    rank_value,
    register_all,
)
from lieutenant_of_poker.rules_engine import (
    RuleContext,
    RulesEngine,
    TrackedHand,
    TrackedAction,
    Severity,
)
from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.action_detector import PlayerAction
from lieutenant_of_poker.card_detector import Card, Rank, Suit


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_pair_true(self):
        """Test pair detection."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
        ]
        assert is_pair(cards) is True

    def test_is_pair_false(self):
        """Test non-pair detection."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
        ]
        assert is_pair(cards) is False

    def test_is_pair_wrong_count(self):
        """Test pair with wrong number of cards."""
        cards = [Card(Rank.ACE, Suit.HEARTS)]
        assert is_pair(cards) is False

    def test_is_suited_true(self):
        """Test suited detection."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
        ]
        assert is_suited(cards) is True

    def test_is_suited_false(self):
        """Test offsuit detection."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
        ]
        assert is_suited(cards) is False

    def test_get_high_cards(self):
        """Test getting high cards sorted."""
        cards = [
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
        ]
        high, low = get_high_cards(cards)
        assert high == Rank.ACE
        assert low == Rank.FIVE

    def test_rank_value(self):
        """Test rank value ordering."""
        assert rank_value(Rank.TWO) < rank_value(Rank.THREE)
        assert rank_value(Rank.KING) < rank_value(Rank.ACE)
        assert rank_value(Rank.TEN) < rank_value(Rank.JACK)

    def test_is_premium_hand_aces(self):
        """Test AA is premium."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.SPADES),
        ]
        assert is_premium_hand(cards) is True

    def test_is_premium_hand_ak(self):
        """Test AK is premium."""
        cards = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES),
        ]
        assert is_premium_hand(cards) is True

    def test_is_premium_hand_72(self):
        """Test 72o is not premium."""
        cards = [
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.TWO, Suit.SPADES),
        ]
        assert is_premium_hand(cards) is False

    def test_is_premium_hand_tens(self):
        """Test TT is premium."""
        cards = [
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.TEN, Suit.SPADES),
        ]
        assert is_premium_hand(cards) is True

    def test_is_premium_hand_nines(self):
        """Test 99 is not premium (just below threshold)."""
        cards = [
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.NINE, Suit.SPADES),
        ]
        assert is_premium_hand(cards) is False


class TestFoldWhenCanCheckRule:
    """Tests for FoldWhenCanCheckRule."""

    @pytest.fixture
    def rule(self):
        return FoldWhenCanCheckRule()

    @pytest.fixture
    def context(self):
        state = GameState(street=Street.FLOP)
        hand = TrackedHand(hand_id="1")
        return RuleContext(current_hand=hand, state=state)

    def test_no_violation_when_not_folding(self, rule, context):
        """No violation when not folding."""
        context.action_about_to_take = PlayerAction.CALL
        context.available_actions = [PlayerAction.FOLD, PlayerAction.CALL]

        violation = rule.evaluate(context)
        assert violation is None

    def test_no_violation_when_fold_is_only_option(self, rule, context):
        """No violation when check not available."""
        context.action_about_to_take = PlayerAction.FOLD
        context.available_actions = [PlayerAction.FOLD, PlayerAction.CALL]

        violation = rule.evaluate(context)
        assert violation is None

    def test_violation_when_folding_with_check_available(self, rule, context):
        """Violation when folding but check is available."""
        context.action_about_to_take = PlayerAction.FOLD
        context.available_actions = [PlayerAction.FOLD, PlayerAction.CHECK, PlayerAction.BET]

        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.severity == Severity.ERROR
        assert "check" in violation.message.lower() or "check" in violation.suggestion.lower()


class TestOpenLimpRule:
    """Tests for OpenLimpRule."""

    @pytest.fixture
    def rule(self):
        return OpenLimpRule()

    def test_no_violation_postflop(self, rule):
        """No violation after flop."""
        state = GameState(street=Street.FLOP)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_no_violation_when_facing_raise(self, rule):
        """No violation when calling a raise."""
        state = GameState(street=Street.PREFLOP)
        hand = TrackedHand(
            hand_id="1",
            actions=[
                TrackedAction(
                    position="BTN",
                    action=PlayerAction.RAISE,
                    amount=100,
                    street=Street.PREFLOP,
                )
            ],
        )
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_violation_on_open_limp(self, rule):
        """Violation when open-limping preflop."""
        state = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.SEVEN, Suit.HEARTS),
                Card(Rank.TWO, Suit.SPADES),
            ],
        )
        hand = TrackedHand(hand_id="1", actions=[])  # No prior raises
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.severity == Severity.WARNING
        assert "limp" in violation.message.lower()


class TestPremiumHandLimpRule:
    """Tests for PremiumHandLimpRule."""

    @pytest.fixture
    def rule(self):
        return PremiumHandLimpRule()

    def test_no_violation_postflop(self, rule):
        """No violation after flop."""
        state = GameState(
            street=Street.FLOP,
            hero_cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.ACE, Suit.SPADES),
            ],
        )
        hand = TrackedHand(hand_id="1")
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_no_violation_with_weak_hand(self, rule):
        """No violation when limping with weak hand."""
        state = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.SEVEN, Suit.HEARTS),
                Card(Rank.TWO, Suit.SPADES),
            ],
        )
        hand = TrackedHand(hand_id="1")
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_violation_limping_aces(self, rule):
        """Violation when limping with AA."""
        state = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.ACE, Suit.SPADES),
            ],
        )
        hand = TrackedHand(hand_id="1")
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.severity == Severity.WARNING
        assert "premium" in violation.message.lower()

    def test_violation_limping_ak(self, rule):
        """Violation when limping with AK."""
        state = GameState(
            street=Street.PREFLOP,
            hero_cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.KING, Suit.SPADES),
            ],
        )
        hand = TrackedHand(hand_id="1")
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is not None


class TestShortStackRule:
    """Tests for ShortStackRule."""

    @pytest.fixture
    def rule(self):
        return ShortStackRule()

    def test_no_violation_postflop(self, rule):
        """No violation after flop."""
        state = GameState(street=Street.FLOP)
        hand = TrackedHand(hand_id="1", hero_chips=500, big_blind=100)
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_no_violation_with_deep_stack(self, rule):
        """No violation with 20 BB stack."""
        state = GameState(street=Street.PREFLOP)
        hand = TrackedHand(hand_id="1", hero_chips=2000, big_blind=100)
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is None

    def test_violation_calling_short_stacked(self, rule):
        """Violation when calling with 5 BB."""
        state = GameState(street=Street.PREFLOP)
        hand = TrackedHand(hand_id="1", hero_chips=500, big_blind=100)
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.CALL,
        )

        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.severity == Severity.WARNING
        assert "5.0" in violation.message or "5 BB" in violation.message

    def test_info_raising_short_stacked(self, rule):
        """Info when raising with 8 BB."""
        state = GameState(street=Street.PREFLOP)
        hand = TrackedHand(hand_id="1", hero_chips=800, big_blind=100)
        context = RuleContext(
            current_hand=hand,
            state=state,
            action_about_to_take=PlayerAction.RAISE,
        )

        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.severity == Severity.INFO


class TestRegisterAll:
    """Tests for register_all function."""

    def test_register_all(self):
        """Test that register_all registers rules."""
        engine = RulesEngine()
        register_all(engine)

        rules = engine.list_rules()
        rule_names = [name for name, _, _ in rules]

        assert "fold_when_can_check" in rule_names
        assert "open_limp" in rule_names
        assert "premium_limp" in rule_names
        assert "short_stack" in rule_names
