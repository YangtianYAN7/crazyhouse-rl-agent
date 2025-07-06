import chess
import chess.variant

all_legal_moves = list({move.uci() for board in [chess.variant.CrazyhouseBoard()] for move in board.legal_moves})
uci_to_index = {uci: i for i, uci in enumerate(all_legal_moves)}
index_to_uci = {i: uci for uci, i in uci_to_index.items()}

def encode_action(uci):
    return uci_to_index.get(uci, 0)

def decode_action(index):
    return index_to_uci.get(index, "0000")















    
