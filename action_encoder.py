PIECES = ['P', 'N', 'B', 'R', 'Q']  

def get_all_possible_actions():
    actions = []

    for from_x in range(8):
        for from_y in range(8):
            for to_x in range(8):
                for to_y in range(8):
                    if (from_x, from_y) != (to_x, to_y):
                        actions.append(('move', (from_x, from_y), (to_x, to_y)))

    for piece in PIECES:
        for x in range(8):
            for y in range(8):
                actions.append(('drop', piece, (x, y)))
    return actions

ALL_ACTIONS = get_all_possible_actions()
ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ALL_ACTIONS)}
INDEX_TO_ACTION = {idx: action for idx, action in enumerate(ALL_ACTIONS)}

def encode_action(action):
    return ACTION_TO_INDEX.get(action, -1)  

def decode_index(index):
    return INDEX_TO_ACTION.get(index, None)

def legal_action_indices(legal_actions):
    return [encode_action(a) for a in legal_actions if encode_action(a) >= 0]

if __name__ == '__main__':
    print(f"Total encoded actions: {len(ALL_ACTIONS)}")
    sample = ('drop', 'N', (3, 4))
    idx = encode_action(sample)
    print(f"Action {sample} -> Index {idx}")
    print(f"Index {idx} -> Action {decode_index(idx)}")
    
