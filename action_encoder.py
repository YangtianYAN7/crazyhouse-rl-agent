# action_encoder.py - 支持按需生成是否含 drop 的动作集

import chess

def get_all_moves(allow_drops=True):
    moves = []

    if allow_drops:
        drop_pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for pt in drop_pieces:
            for to_sq in range(64):
                rank = chess.square_rank(to_sq)
                if pt == chess.PAWN and (rank == 0 or rank == 7):
                    continue
                drop_move = chess.Move(from_square=None, to_square=to_sq, drop=pt)
                moves.append(drop_move.uci())

    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            moves.append(move.uci())
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                moves.append(promo_move.uci())

    moves.append("0000")
    return moves

ALL_POSSIBLE_MOVES = get_all_moves(allow_drops=True)

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























































    
