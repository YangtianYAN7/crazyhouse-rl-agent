# action_encoder.py

import chess

# 构造包含升变的固定动作集合
ALL_POSSIBLE_MOVES = []

promotion_pieces = ['q', 'r', 'b', 'n']

for from_square in range(64):
    for to_square in range(64):
        if from_square == to_square:
            continue
        move = chess.Move(from_square, to_square)
        move_str = move.uci()
        ALL_POSSIBLE_MOVES.append(move_str)

        # 添加升变走法（仅在第7排到8排或2排到1排才有升变可能）
        if chess.square_rank(from_square) == 6 and chess.square_rank(to_square) == 7:
            for promo in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(promo.upper()).piece_type)
                ALL_POSSIBLE_MOVES.append(move.uci())
        if chess.square_rank(from_square) == 1 and chess.square_rank(to_square) == 0:
            for promo in promotion_pieces:
                move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(promo.upper()).piece_type)
                ALL_POSSIBLE_MOVES.append(move.uci())

# 添加一个 fallback 空动作
ALL_POSSIBLE_MOVES.append("0000")

def encode_action(board, legal_moves):
    return ALL_POSSIBLE_MOVES

def decode_action(board, action_index):
    legal_moves = list(board.legal_moves)
    move_str = ALL_POSSIBLE_MOVES[action_index]
    for move in legal_moves:
        if move.uci() == move_str:
            return move
    return legal_moves[0] if legal_moves else chess.Move.null()

def encode_action_index(board, move):
    return ALL_POSSIBLE_MOVES.index(move.uci())






























    
