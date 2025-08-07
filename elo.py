# elo.py - 简易 Elo 评估系统

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, result, k=32):
    """
    result: 1=win for A, 0=loss for A, 0.5=draw
    """
    expected_a = expected_score(elo_a, elo_b)
    expected_b = expected_score(elo_b, elo_a)

    new_elo_a = elo_a + k * (result - expected_a)
    new_elo_b = elo_b + k * ((1 - result) - expected_b)

    return new_elo_a, new_elo_b

