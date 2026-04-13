"""Quick diagnostic: does extract_game_state_at_prefix return correct scores at 99%?"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, '.')
from pitch_sequence_predictor import build_vocab_maps, CONTINUOUS_PITCH_COLS
from evaluate_win_probability import extract_game_state_at_prefix

tp = pd.read_csv('test_pitch_2025-08-01_2025-11-03.csv')
tpc = pd.read_csv('test_pitch_context_2025-08-01_2025-11-03.csv')
tpr = pd.read_csv('test_pitch_result_2025-08-01_2025-11-03.csv')
tab = pd.read_csv('test_at_bat_target_2025-08-01_2025-11-03.csv')
tgt = pd.read_csv('test_Game_target_2025-08-01_2025-11-03.csv')
tgc = pd.read_csv('test_game_context_2025-08-01_2025-11-03.csv')

pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx = build_vocab_maps()
pitch_mean = np.zeros(len(CONTINUOUS_PITCH_COLS))
pitch_std = np.ones(len(CONTINUOUS_PITCH_COLS))

game_ids = tgc['game_id'].values[:20]
n_correct = 0
n_total = 0
for i, gid in enumerate(game_ids):
    state = extract_game_state_at_prefix(
        gid, 0.99, tp, tpc, tpr, tab,
        pt_to_idx, pr_to_idx, ev_to_idx, zone_to_idx,
        pitch_mean, pitch_std
    )
    actual = tgt['home_win_exp'].iloc[i]
    if state is None:
        print(f"  {gid}: state=None")
        continue
    n_total += 1
    h, a = state['home_score'], state['away_score']
    prefix_winner = 1 if h > a else (0 if h < a else 0.5)
    match = (prefix_winner == actual)
    if match:
        n_correct += 1
    print(f"  {gid}: H={h} A={a} inn={state['inning']} top={state['is_top']} "
          f"outs={state['outs']} pitches={state['n_pitches_replayed']}/{state['total_pitches']} "
          f"prefix_winner={'H' if h>a else 'A' if a>h else 'T'} actual={actual} {'OK' if match else 'WRONG'}")

print(f"\nPrefix score matches actual winner: {n_correct}/{n_total} = {n_correct/n_total:.1%}")
