"""Tests for rules engine module."""

import pytest
from unittest.mock import MagicMock

from lieutenant_of_poker.rules_engine import (
    Rule,
    RuleContext,
    RuleViolation,
    RulesEngine,
    Severity,
    TrackedHand,
    TrackedAction,
    create_rule,
)
from lieutenant_of_poker.game_state import GameState, Street
from lieutenant_of_poker.action_detector import PlayerAction


class TestSeverity:
    """Tests for Severity enum."""

    def test_severity_ordering(self):
        """Test severity values are ordered correctly."""
        assert Severity.INFO.value < Severity.WARNING.value
        assert Severity.WARNING.value < Severity.ERROR.value
        assert Severity.ERROR.value < Severity.CRITICAL.value


class TestRuleViolation:
    """Tests for RuleViolation dataclass."""

    def test_violation_creation(self):
        """Test creating a violation."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
            suggestion="Test suggestion",
            details={"key": "value"},
        )
        assert violation.rule_name == "test_rule"
        assert violation.severity == Severity.WARNING
        assert violation.message == "Test message"
        assert violation.suggestion == "Test suggestion"
        assert violation.details == {"key": "value"}

    def test_violation_str(self):
        """Test string representation of violation."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.ERROR,
            message="Bad play detected",
            suggestion="Consider folding",
        )
        result = str(violation)
        assert "[ERROR]" in result
        assert "test_rule" in result
        assert "Bad play detected" in result
        assert "Consider folding" in result

    def test_violation_str_no_suggestion(self):
        """Test string representation without suggestion."""
        violation = RuleViolation(
            rule_name="test_rule",
            severity=Severity.INFO,
            message="Info message",
        )
        result = str(violation)
        assert "[INFO]" in result
        assert "Suggestion" not in result


class TestTrackedHand:
    """Tests for TrackedHand dataclass."""

    def test_hero_stack_bb(self):
        """Test stack in big blinds calculation."""
        hand = TrackedHand(
            hand_id="1",
            hero_chips=1000,
            big_blind=100,
        )
        assert hand.hero_stack_bb == 10.0

    def test_hero_stack_bb_zero_blind(self):
        """Test stack in big blinds with zero blind."""
        hand = TrackedHand(
            hand_id="1",
            hero_chips=1000,
            big_blind=0,
        )
        assert hand.hero_stack_bb == 0.0


class TestRuleContext:
    """Tests for RuleContext dataclass."""

    def test_is_preflop(self):
        """Test preflop detection."""
        state = GameState(street=Street.PREFLOP)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)
        assert context.is_preflop is True

    def test_is_not_preflop(self):
        """Test non-preflop detection."""
        state = GameState(street=Street.FLOP)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)
        assert context.is_preflop is False

    def test_pot_from_state(self):
        """Test pot value from state."""
        state = GameState(pot=500)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)
        assert context.pot == 500

    def test_pot_none_returns_zero(self):
        """Test pot returns 0 when None."""
        state = GameState(pot=None)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)
        assert context.pot == 0


class SampleRule(Rule):
    """A sample rule for testing."""

    name = "sample_rule"
    description = "A sample rule for testing"
    severity = Severity.WARNING

    def __init__(self, should_violate: bool = False):
        self.should_violate = should_violate

    def evaluate(self, context: RuleContext) -> RuleViolation | None:
        if self.should_violate:
            return self._create_violation(
                message="Sample violation",
                suggestion="Fix it",
            )
        return None


class ErrorRule(Rule):
    """A rule that raises an error."""

    name = "error_rule"
    description = "A rule that errors"
    severity = Severity.ERROR

    def evaluate(self, context: RuleContext) -> RuleViolation | None:
        raise ValueError("Rule evaluation failed")


class TestRule:
    """Tests for Rule base class."""

    def test_rule_evaluate_no_violation(self):
        """Test rule evaluation with no violation."""
        rule = SampleRule(should_violate=False)
        state = GameState()
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)

        result = rule.evaluate(context)
        assert result is None

    def test_rule_evaluate_with_violation(self):
        """Test rule evaluation with violation."""
        rule = SampleRule(should_violate=True)
        state = GameState()
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)

        result = rule.evaluate(context)
        assert result is not None
        assert result.rule_name == "sample_rule"
        assert result.severity == Severity.WARNING
        assert "Sample violation" in result.message

    def test_create_violation_helper(self):
        """Test _create_violation helper method."""
        rule = SampleRule()
        violation = rule._create_violation(
            message="Test message",
            suggestion="Test suggestion",
            severity=Severity.ERROR,
            extra_key="extra_value",
        )
        assert violation.rule_name == "sample_rule"
        assert violation.severity == Severity.ERROR  # Override default
        assert violation.message == "Test message"
        assert violation.suggestion == "Test suggestion"
        assert violation.details["extra_key"] == "extra_value"


class TestRulesEngine:
    """Tests for RulesEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a rules engine for testing."""
        return RulesEngine()

    @pytest.fixture
    def context(self):
        """Create a rule context for testing."""
        state = GameState()
        hand = TrackedHand(hand_id="1")
        return RuleContext(current_hand=hand, state=state)

    def test_register_rule(self, engine):
        """Test registering a rule."""
        rule = SampleRule()
        engine.register_rule(rule)

        assert "sample_rule" in [name for name, _, _ in engine.list_rules()]
        assert engine.get_rule("sample_rule") == rule

    def test_unregister_rule(self, engine):
        """Test unregistering a rule."""
        rule = SampleRule()
        engine.register_rule(rule)

        result = engine.unregister_rule("sample_rule")
        assert result is True
        assert engine.get_rule("sample_rule") is None

    def test_unregister_nonexistent_rule(self, engine):
        """Test unregistering a rule that doesn't exist."""
        result = engine.unregister_rule("nonexistent")
        assert result is False

    def test_enable_rule(self, engine):
        """Test enabling a rule."""
        rule = SampleRule()
        engine.register_rule(rule)
        engine.disable_rule("sample_rule")

        result = engine.enable_rule("sample_rule")
        assert result is True

        rules = engine.list_rules()
        assert ("sample_rule", True, rule.description) in rules

    def test_enable_nonexistent_rule(self, engine):
        """Test enabling a rule that doesn't exist."""
        result = engine.enable_rule("nonexistent")
        assert result is False

    def test_disable_rule(self, engine):
        """Test disabling a rule."""
        rule = SampleRule()
        engine.register_rule(rule)

        result = engine.disable_rule("sample_rule")
        assert result is True

        rules = engine.list_rules()
        assert ("sample_rule", False, rule.description) in rules

    def test_disable_nonexistent_rule(self, engine):
        """Test disabling a rule that doesn't exist."""
        result = engine.disable_rule("nonexistent")
        assert result is False

    def test_enable_all(self, engine):
        """Test enabling all rules."""
        engine.register_rule(SampleRule())
        engine.register_rule(ErrorRule())
        engine.disable_all()

        engine.enable_all()

        rules = engine.list_rules()
        assert all(enabled for _, enabled, _ in rules)

    def test_disable_all(self, engine):
        """Test disabling all rules."""
        engine.register_rule(SampleRule())
        engine.register_rule(ErrorRule())

        engine.disable_all()

        rules = engine.list_rules()
        assert all(not enabled for _, enabled, _ in rules)

    def test_evaluate_all_no_violations(self, engine, context):
        """Test evaluate_all with no violations."""
        engine.register_rule(SampleRule(should_violate=False))

        violations = engine.evaluate_all(context)
        assert violations == []

    def test_evaluate_all_with_violations(self, engine, context):
        """Test evaluate_all with violations."""
        engine.register_rule(SampleRule(should_violate=True))

        violations = engine.evaluate_all(context)
        assert len(violations) == 1
        assert violations[0].rule_name == "sample_rule"

    def test_evaluate_all_skips_disabled(self, engine, context):
        """Test evaluate_all skips disabled rules."""
        rule = SampleRule(should_violate=True)
        engine.register_rule(rule)
        engine.disable_rule("sample_rule")

        violations = engine.evaluate_all(context)
        assert violations == []

    def test_evaluate_all_handles_errors(self, engine, context):
        """Test evaluate_all handles rule errors gracefully."""
        engine.register_rule(ErrorRule())

        violations = engine.evaluate_all(context)
        assert len(violations) == 1
        assert "error" in violations[0].message.lower()

    def test_evaluate_by_severity(self, engine, context):
        """Test filtering violations by severity."""
        # Create rules with different severities
        info_rule = SampleRule(should_violate=True)
        info_rule.name = "info_rule"
        info_rule.severity = Severity.INFO

        warning_rule = SampleRule(should_violate=True)
        warning_rule.name = "warning_rule"
        warning_rule.severity = Severity.WARNING

        error_rule = SampleRule(should_violate=True)
        error_rule.name = "error_rule"
        error_rule.severity = Severity.ERROR

        engine.register_rule(info_rule)
        engine.register_rule(warning_rule)
        engine.register_rule(error_rule)

        # Filter by WARNING and above
        violations = engine.evaluate_by_severity(context, Severity.WARNING)
        assert len(violations) == 2
        names = [v.rule_name for v in violations]
        assert "warning_rule" in names
        assert "error_rule" in names
        assert "info_rule" not in names


class TestCreateRule:
    """Tests for create_rule factory function."""

    def test_create_rule_from_function(self):
        """Test creating a rule from a function."""

        def my_evaluate(context: RuleContext) -> RuleViolation | None:
            if context.pot > 100:
                return RuleViolation(
                    rule_name="big_pot_rule",
                    severity=Severity.INFO,
                    message="Pot is large",
                )
            return None

        rule = create_rule(
            name="big_pot_rule",
            evaluate_fn=my_evaluate,
            description="Warns about large pots",
            severity=Severity.INFO,
        )

        assert rule.name == "big_pot_rule"
        assert rule.description == "Warns about large pots"

        # Test with small pot
        state = GameState(pot=50)
        hand = TrackedHand(hand_id="1")
        context = RuleContext(current_hand=hand, state=state)
        assert rule.evaluate(context) is None

        # Test with large pot
        state = GameState(pot=150)
        context = RuleContext(current_hand=hand, state=state)
        violation = rule.evaluate(context)
        assert violation is not None
        assert violation.message == "Pot is large"
