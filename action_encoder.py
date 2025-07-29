# action_encoder.py - 完整包含所有 Crazyhouse 动作，无需提前合法性判断

import chess

ALL_POSSIBLE_MOVES = []

# 构造 drop 动作（P/N/B/R/Q @ 任意可落子格）
drop_pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
for pt in drop_pieces:
    for to_sq in range(64):
        # 跳过兵掉到第1/第8排（实际对局规则禁止）
        rank = chess.square_rank(to_sq)
        if pt == chess.PAWN and (rank == 0 or rank == 7):
            continue
        drop_move = chess.Move(from_square=None, to_square=to_sq, drop=pt)
        ALL_POSSIBLE_MOVES.append(drop_move.uci())

# 构造所有普通走子 + 升变
for from_sq in range(64):
    for to_sq in range(64):
        if from_sq == to_sq:
            continue
        # 普通走子
        move = chess.Move(from_sq, to_sq)
        ALL_POSSIBLE_MOVES.append(move.uci())
        # 升变走子
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promo_move = chess.Move(from_sq, to_sq, promotion=promo)
            ALL_POSSIBLE_MOVES.append(promo_move.uci())

# fallback 空动作
ALL_POSSIBLE_MOVES.append("0000")

def encode_action(board, legal_moves):
    return ALL_POSSIBLE_MOVES

def decode_action(board, action_index):
    move_str = ALL_POSSIBLE_MOVES[action_index]
    try:
        return chess.Move.from_uci(move_str)
    except:
        return chess.Move.null()

def encode_action_index(board, move):
    return ALL_POSSIBLE_MOVES.index(move.uci())





















































    
