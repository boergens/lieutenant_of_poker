"""
Player seat positions for Governor of Poker table.

Positions are (x, y) coordinates of the top corner of the currency symbol
for each seat.
"""

SEAT_POSITIONS = (
    (632, 155),   # Seat 0: top center
    (1153, 257),  # Seat 1: top right
    (1166, 535),  # Seat 2: right
    (846, 710),   # Seat 3: bottom right
    (418, 710),   # Seat 4: bottom left
    (124, 535),   # Seat 5: left
    (151, 257),   # Seat 6: top left
)

# Blind indicator positions - where blind amounts appear next to each seat
# None means no blind position known for that seat
# Same offset applies as for player money (_MONEY_OFFSET_X, _MONEY_OFFSET_Y)
BLIND_POSITIONS: tuple[tuple[int, int] | None, ...] = (
    (643, 249),   # Seat 0: top center
    (964, 295),   # Seat 1: top right
    (977, 480),   # Seat 2: right
    (857, 572),   # Seat 3: bottom right
    (430, 572),   # Seat 4: bottom left
    (336, 480),   # Seat 5: left
    (363, 295),   # Seat 6: top left
)
