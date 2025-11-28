"""
Pluggable rules engine for poker mistake detection.

Provides an extensible framework for defining and evaluating poker rules
against game states to detect strategic mistakes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol

from .game_state import GameState, Street
from .action_detector import PlayerAction, DetectedAction
from .card_detector import Card


class Severity(Enum):
    """Severity levels for rule violations."""

    INFO = auto()  # Informational, not necessarily a mistake
    WARNING = auto()  # Suboptimal play, minor mistake
    ERROR = auto()  # Clear mistake, significant EV loss
    CRITICAL = auto()  # Major mistake, likely losing play


@dataclass
class RuleViolation:
    """A detected rule violation."""

    rule_name: str
    severity: Severity
    message: str
    suggestion: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.severity.name}] {self.rule_name}: {self.message}"]
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class TrackedAction:
    """An action tracked during a hand."""

    position: str
    action: PlayerAction
    amount: Optional[int] = None
    street: Street = Street.UNKNOWN
    timestamp_ms: float = 0.0


@dataclass
class TrackedHand:
    """A poker hand being tracked."""

    hand_id: str
    hero_cards: list[Card] = field(default_factory=list)
    community_cards: list[Card] = field(default_factory=list)
    actions: list[TrackedAction] = field(default_factory=list)
    pot: int = 0
    hero_chips: int = 0
    current_street: Street = Street.PREFLOP
    is_hero_turn: bool = False
    hero_position: str = "BOTTOM"
    big_blind: int = 0

    @property
    def hero_stack_bb(self) -> float:
        """Get hero stack in big blinds."""
        if self.big_blind <= 0:
            return 0.0
        return self.hero_chips / self.big_blind

    @property
    def pot_odds(self) -> float:
        """Get current pot odds as a ratio."""
        # This would need the current bet to call
        return 0.0


@dataclass
class RuleContext:
    """Context provided to rules for evaluation."""

    current_hand: TrackedHand
    state: GameState
    available_actions: list[PlayerAction] = field(default_factory=list)
    action_about_to_take: Optional[PlayerAction] = None
    amount_to_call: int = 0

    @property
    def is_preflop(self) -> bool:
        """Check if we're preflop."""
        return self.state.street == Street.PREFLOP

    @property
    def hero_cards(self) -> list[Card]:
        """Get hero's hole cards."""
        return self.state.hero_cards

    @property
    def community_cards(self) -> list[Card]:
        """Get community cards."""
        return self.state.community_cards

    @property
    def pot(self) -> int:
        """Get current pot size."""
        return self.state.pot or 0

    @property
    def hero_chips(self) -> int:
        """Get hero's chip count."""
        return self.state.hero_chips or 0


class Rule(ABC):
    """Abstract base class for poker rules."""

    name: str = "unnamed_rule"
    description: str = ""
    severity: Severity = Severity.WARNING

    @abstractmethod
    def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
        """
        Evaluate the rule against the current game context.

        Args:
            context: The current game context including state and hand history.

        Returns:
            RuleViolation if the rule is broken, None otherwise.
        """
        pass

    def _create_violation(
        self,
        message: str,
        suggestion: Optional[str] = None,
        severity: Optional[Severity] = None,
        **details: Any,
    ) -> RuleViolation:
        """Helper to create a violation for this rule."""
        return RuleViolation(
            rule_name=self.name,
            severity=severity or self.severity,
            message=message,
            suggestion=suggestion,
            details=details,
        )


class SolverClient(Protocol):
    """Protocol for solver integration (future use)."""

    def get_optimal_action(
        self,
        hero_cards: list[Card],
        community_cards: list[Card],
        pot: int,
        stack: int,
        position: str,
        action_sequence: list[str],
    ) -> dict[str, Any]:
        """Get the optimal action from a solver."""
        ...


class RulesEngine:
    """Manages and executes rules against game states."""

    def __init__(self):
        """Initialize the rules engine."""
        self._rules: dict[str, Rule] = {}
        self._enabled_rules: set[str] = set()

    def register_rule(self, rule: Rule) -> None:
        """
        Register a rule with the engine.

        Args:
            rule: The rule to register.
        """
        self._rules[rule.name] = rule
        self._enabled_rules.add(rule.name)

    def unregister_rule(self, rule_name: str) -> bool:
        """
        Unregister a rule from the engine.

        Args:
            rule_name: Name of the rule to remove.

        Returns:
            True if rule was removed, False if not found.
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            self._enabled_rules.discard(rule_name)
            return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """
        Enable a registered rule.

        Args:
            rule_name: Name of the rule to enable.

        Returns:
            True if rule was enabled, False if not found.
        """
        if rule_name in self._rules:
            self._enabled_rules.add(rule_name)
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """
        Disable a registered rule.

        Args:
            rule_name: Name of the rule to disable.

        Returns:
            True if rule was disabled, False if not found.
        """
        if rule_name in self._rules:
            self._enabled_rules.discard(rule_name)
            return True
        return False

    def enable_all(self) -> None:
        """Enable all registered rules."""
        self._enabled_rules = set(self._rules.keys())

    def disable_all(self) -> None:
        """Disable all rules."""
        self._enabled_rules.clear()

    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """Get a rule by name."""
        return self._rules.get(rule_name)

    def list_rules(self) -> list[tuple[str, bool, str]]:
        """
        List all registered rules.

        Returns:
            List of (name, enabled, description) tuples.
        """
        return [
            (name, name in self._enabled_rules, rule.description)
            for name, rule in self._rules.items()
        ]

    def evaluate_all(self, context: RuleContext) -> list[RuleViolation]:
        """
        Evaluate all enabled rules against the context.

        Args:
            context: The game context to evaluate.

        Returns:
            List of rule violations detected.
        """
        violations = []

        for rule_name in self._enabled_rules:
            rule = self._rules.get(rule_name)
            if rule is None:
                continue

            try:
                violation = rule.evaluate(context)
                if violation is not None:
                    violations.append(violation)
            except Exception as e:
                # Log error but continue with other rules
                violations.append(
                    RuleViolation(
                        rule_name=rule_name,
                        severity=Severity.INFO,
                        message=f"Rule evaluation error: {e}",
                    )
                )

        return violations

    def evaluate_by_severity(
        self, context: RuleContext, min_severity: Severity = Severity.WARNING
    ) -> list[RuleViolation]:
        """
        Evaluate rules and filter by minimum severity.

        Args:
            context: The game context to evaluate.
            min_severity: Minimum severity to include.

        Returns:
            List of violations meeting the severity threshold.
        """
        all_violations = self.evaluate_all(context)
        return [v for v in all_violations if v.severity.value >= min_severity.value]


def create_rule(
    name: str,
    evaluate_fn: Callable[[RuleContext], Optional[RuleViolation]],
    description: str = "",
    severity: Severity = Severity.WARNING,
) -> Rule:
    """
    Create a rule from a function.

    Args:
        name: Rule name.
        evaluate_fn: Function that evaluates the rule.
        description: Rule description.
        severity: Default severity for violations.

    Returns:
        A Rule instance.
    """

    class FunctionalRule(Rule):
        def evaluate(self, context: RuleContext) -> Optional[RuleViolation]:
            return evaluate_fn(context)

    rule = FunctionalRule()
    rule.name = name
    rule.description = description
    rule.severity = severity
    return rule
