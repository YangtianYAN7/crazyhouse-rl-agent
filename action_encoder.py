import chess.variant

ALL_SQUARES = [chess.square(file, rank) for rank in range(8) for file in range(8)]
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
ACTION_TO_INDEX = {}
INDEX_TO_ACTION = {}
index = 0

# 普通走子动作（from_square -> to_square）
for from_square in ALL_SQUARES:
    for to_square in ALL_SQUARES:
        if from_square != to_square:
            ACTION_TO_INDEX[("move", from_square, to_square)] = index
            INDEX_TO_ACTION[index] = ("move", from_square, to_square)
            index += 1

# 投子动作（drop piece -> to_square）
for piece_type in PIECE_TYPES:
    for square in ALL_SQUARES:
        ACTION_TO_INDEX[("drop", piece_type, square)] = index
        INDEX_TO_ACTION[index] = ("drop", piece_type, square)
        index += 1

NUM_ACTIONS = index

def encode_action(move, board):
    if isinstance(move, chess.Move) and move.drop is not None:
        # 投子动作
        return ACTION_TO_INDEX.get(("drop", move.drop, move.to_square), -1)
    else:
        # 普通走子动作
        return ACTION_TO_INDEX.get(("move", move.from_square, move.to_square), -1)

def decode_action(index, board):
    action = INDEX_TO_ACTION.get(index)
    if action is None:
        return None
    kind, a, b = action
    if kind == "move":
        return chess.Move(a, b)
    elif kind == "drop":
        return chess.Move.drop(a, b)








    
