import pandas as pd
pc = pd.read_csv('test_pitch_context_2025-08-01_2025-11-03.csv')
pitch = pd.read_csv('test_pitch_2025-08-01_2025-11-03.csv')

print(f"test_pitch len: {len(pitch)}, test_pc len: {len(pc)}")
print(f"test_pitch has game_id: {'game_id' in pitch.columns}")
print(f"test_pc has game_id: {'game_id' in pc.columns}")

# Simulate what the code does: subset to 2% of games
all_test_ids = pitch['game_id'].unique()
m_games = max(1, int(len(all_test_ids) * 0.02))
keep_test = set(all_test_ids[-m_games:])
print(f"\nKeeping {m_games} games: {keep_test}")

mask = pitch['game_id'].isin(keep_test)
test_pitch = pitch[mask].reset_index(drop=True)
test_pc = pc[mask].reset_index(drop=True)

print(f"After subset: test_pitch len={len(test_pitch)}, test_pc len={len(test_pc)}")

# Now check what happens when we look up scores
for gid in keep_test:
    gm = test_pitch['game_id'] == gid
    hs = test_pc.loc[gm, 'home_score'].max()
    aws = test_pc.loc[gm, 'away_score'].max()
    
    # Also check directly from test_pc
    gm2 = test_pc['game_id'] == gid
    hs2 = test_pc.loc[gm2, 'home_score'].max()
    aws2 = test_pc.loc[gm2, 'away_score'].max()
    
    print(f"Game {gid}:")
    print(f"  Via test_pitch mask: home={hs}, away={aws}")
    print(f"  Via test_pc mask:    home={hs2}, away={aws2}")
