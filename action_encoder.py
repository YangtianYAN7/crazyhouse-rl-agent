import chess

def encode_action(move, board):

    if isinstance(move, chess.Move) and move.drop is not None:
        # Drop move
        piece_type = move.drop  # 1–6
        to_square = move.to_square
        return (piece_type - 1) * 64 + to_square
    elif isinstance(move, chess.Move):
        # 普通移动
        from_square = move.from_square
        to_square = move.to_square
        return 384 + from_square * 64 + to_square
    else:
        raise ValueError(f"Unsupported move type: {move}")


def decode_action(index, board):

    if index < 384:
        # Drop move
        piece_type = (index // 64) + 1
        to_square = index % 64
        return chess.Drop(piece_type, to_square)
    else:
        # 普通 move
        from_square = (index - 384) // 64
        to_square = (index - 384) % 64
        return chess.Move(from_square, to_square)






    
