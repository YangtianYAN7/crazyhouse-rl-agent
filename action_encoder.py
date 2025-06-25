import chess

def encode_action(move: chess.Move) -> int:
    """
    将 chess.Move 编码为整数索引。
    编码方式：from_square * 64 + to_square （最多 64*64 = 4096 种动作）
    """
    return move.from_square * 64 + move.to_square

def decode_action(action_index: int) -> chess.Move:
    """
    将整数动作索引解码为 chess.Move。
    解码方式与 encode_action 对应。
    """
    from_square = action_index // 64
    to_square = action_index % 64
    return chess.Move(from_square, to_square)


    
