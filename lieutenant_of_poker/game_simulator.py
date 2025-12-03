"""
Game simulator for completing hands when hero folds.

When the hero folds before showdown, we need to simulate the rest of the hand
to produce valid output for analysis tools. This includes:
- Dealing remaining community cards
- Generating opponent hole cards
- Running a showdown
- Determining a winner
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Union

# Type alias for RNG - either the random module or a Random instance
RNG = Union[random.Random, type(random)]

# All 52 cards in the deck
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['c', 'd', 'h', 's']
FULL_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]


@dataclass
class SimulatedHand:
    """Result of simulating the remainder of a hand."""
    flop: List[str]  # 3 cards
    turn: str  # 1 card
    river: str  # 1 card
    opponent_hands: dict  # player_name -> [card1, card2]
    winner: str  # winning player name
    winning_hand: List[str]  # winner's hole cards


@dataclass
class ShowdownConfig:
    """Configuration for deterministic showdown output.

    Used by export functions to produce deterministic output without RNG.
    Winner is determined by evaluating the hands unless force_winner is set.
    """
    opponent_cards: dict  # player_name -> [card1, card2]
    force_winner: Optional[str] = None  # Override winner (for legacy fixtures)


def get_used_cards(
    hero_cards: List[str],
    community_cards: List[str],
) -> Set[str]:
    """Get all cards already in use."""
    used = set()
    for card in hero_cards:
        if card:
            used.add(card)
    for card in community_cards:
        if card:
            used.add(card)
    return used


def deal_remaining_community(
    current_community: List[str],
    used_cards: Set[str],
    rng: RNG = random,
) -> Tuple[List[str], str, str, Set[str]]:
    """
    Deal remaining community cards to complete through river.

    Returns:
        (flop, turn, river, updated_used_cards)
    """
    available = [c for c in FULL_DECK if c not in used_cards]
    rng.shuffle(available)

    flop = list(current_community[:3]) if len(current_community) >= 3 else []
    turn = current_community[3] if len(current_community) >= 4 else ""
    river = current_community[4] if len(current_community) >= 5 else ""

    idx = 0

    # Complete flop if needed
    while len(flop) < 3:
        flop.append(available[idx])
        used_cards.add(available[idx])
        idx += 1

    # Deal turn if needed
    if not turn:
        turn = available[idx]
        used_cards.add(turn)
        idx += 1

    # Deal river if needed
    if not river:
        river = available[idx]
        used_cards.add(river)
        idx += 1

    return flop, turn, river, used_cards


def deal_opponent_hands(
    opponent_names: List[str],
    used_cards: Set[str],
    rng: RNG = random,
) -> Tuple[dict, Set[str]]:
    """
    Deal hole cards to all opponents.

    Returns:
        (opponent_hands dict, updated_used_cards)
    """
    available = [c for c in FULL_DECK if c not in used_cards]
    rng.shuffle(available)

    opponent_hands = {}
    idx = 0

    for name in opponent_names:
        card1 = available[idx]
        card2 = available[idx + 1]
        opponent_hands[name] = [card1, card2]
        used_cards.add(card1)
        used_cards.add(card2)
        idx += 2

    return opponent_hands, used_cards


def parse_card(card: str) -> Tuple[int, str]:
    """Parse card string like 'Ah' into (rank_value, suit)."""
    rank_str = card[:-1]
    suit = card[-1]
    rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return rank_values.get(rank_str, 0), suit


def evaluate_hand_rank(hole_cards: List[str], community: List[str]) -> Tuple[int, List[int]]:
    """
    Evaluate poker hand strength.

    Returns:
        Tuple of (hand_type, tiebreakers) where:
        - hand_type: 0=high card, 1=pair, 2=two pair, 3=trips, 4=straight,
                     5=flush, 6=full house, 7=quads, 8=straight flush
        - tiebreakers: list of rank values for breaking ties
    """
    from itertools import combinations

    all_cards = hole_cards + community
    if len(all_cards) < 5:
        return (0, [0])

    best_hand = (0, [0])

    # Try all 5-card combinations
    for five_cards in combinations(all_cards, 5):
        parsed = [parse_card(c) for c in five_cards]
        ranks = sorted([r for r, s in parsed], reverse=True)
        suits = [s for r, s in parsed]

        # Check for flush
        is_flush = len(set(suits)) == 1

        # Check for straight
        unique_ranks = sorted(set(ranks), reverse=True)
        is_straight = False
        straight_high = 0
        if len(unique_ranks) == 5:
            if unique_ranks[0] - unique_ranks[4] == 4:
                is_straight = True
                straight_high = unique_ranks[0]
            # Ace-low straight (A-2-3-4-5)
            elif unique_ranks == [14, 5, 4, 3, 2]:
                is_straight = True
                straight_high = 5

        # Count rank frequencies
        rank_counts = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)
        ranks_by_count = sorted(rank_counts.keys(), key=lambda r: (rank_counts[r], r), reverse=True)

        # Determine hand type
        if is_straight and is_flush:
            hand = (8, [straight_high])
        elif counts == [4, 1]:
            hand = (7, ranks_by_count)
        elif counts == [3, 2]:
            hand = (6, ranks_by_count)
        elif is_flush:
            hand = (5, ranks)
        elif is_straight:
            hand = (4, [straight_high])
        elif counts == [3, 1, 1]:
            hand = (3, ranks_by_count)
        elif counts == [2, 2, 1]:
            hand = (2, ranks_by_count)
        elif counts == [2, 1, 1, 1]:
            hand = (1, ranks_by_count)
        else:
            hand = (0, ranks)

        if hand > best_hand:
            best_hand = hand

    return best_hand


def pick_winner(
    all_hands: dict,
    community: List[str],
) -> Tuple[str, List[str]]:
    """
    Pick a winner from all hands using poker hand evaluation.

    Args:
        all_hands: Dict of player_name -> [card1, card2]
        community: List of community cards

    Returns:
        (winner_name, winner_hole_cards)
    """
    if not all_hands:
        raise ValueError("No hands to evaluate")

    best_name = None
    best_rank = (-1, [])

    for name, hole_cards in all_hands.items():
        rank = evaluate_hand_rank(hole_cards, community)
        if rank > best_rank:
            best_rank = rank
            best_name = name

    return best_name, all_hands[best_name]


def simulate_hand_completion(
    hero_cards: List[str],
    community_cards: List[str],
    active_opponents: List[str],
    pot: int,
    rng: RNG = random,
) -> SimulatedHand:
    """
    Simulate the completion of a hand after hero folds.

    Args:
        hero_cards: Hero's hole cards (for avoiding duplicates)
        community_cards: Current community cards (may be incomplete)
        active_opponents: List of opponent names still in the hand
        pot: Current pot size
        rng: Random number generator (default: random module)

    Returns:
        SimulatedHand with all simulated data
    """
    # Get cards already in use
    used_cards = get_used_cards(hero_cards, community_cards)

    # Deal remaining community cards
    flop, turn, river, used_cards = deal_remaining_community(
        community_cards, used_cards, rng
    )

    # Deal opponent hands
    opponent_hands, used_cards = deal_opponent_hands(
        active_opponents, used_cards, rng
    )

    # Pick a winner
    full_community = flop + [turn, river]
    winner, winning_hand = pick_winner(opponent_hands, full_community)

    return SimulatedHand(
        flop=flop,
        turn=turn,
        river=river,
        opponent_hands=opponent_hands,
        winner=winner,
        winning_hand=winning_hand,
    )


def make_showdown_config(
    hero_cards: List[str],
    community_cards: List[str],
    opponent_names: List[str],
    rng: RNG = random,
) -> ShowdownConfig:
    """
    Generate a ShowdownConfig with random opponent cards.

    For CLI use where deterministic output is not required.
    """
    used_cards = get_used_cards(hero_cards, community_cards)
    opponent_hands, _ = deal_opponent_hands(opponent_names, used_cards, rng)
    return ShowdownConfig(opponent_cards=opponent_hands)


def format_showdown_lines(
    hero_name: str,
    hero_cards: List[str],
    opponent_hands: dict,
    include_hero: bool = False,
) -> List[str]:
    """
    Format showdown lines for all players.

    Args:
        hero_name: Name of the hero
        hero_cards: Hero's hole cards
        opponent_hands: Dict of opponent name -> hole cards
        include_hero: Whether to include hero in showdown (usually False if folded)

    Returns:
        List of showdown lines
    """
    lines = []

    if include_hero and hero_cards:
        cards_str = " ".join(hero_cards)
        lines.append(f"Showdown: {hero_name} [{cards_str}]")

    for name, cards in opponent_hands.items():
        cards_str = " ".join(cards)
        lines.append(f"Showdown: {name} [{cards_str}]")

    return lines
