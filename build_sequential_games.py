# this will not be pretty
from tqdm import tqdm
import pandas as pd

post_files = ["mini-normal1.json", "mini-normal2.json", "large-normal.json"]
posts = pd.concat(pd.read_json(fn, orient='records') for fn in post_files)

# filter posts by game into one dataframe per game
games = []
for game_id in posts.game_id.unique():
    game = posts[posts['game_id'] == game_id]
    game.reset_index()
    games.append(game)

# for each game, figure out author labels
slot_files = ["mini-normal-slots.json", "large-normal-slots.json", "old-normal-slots.json"]
slots = pd.concat(pd.read_json(fn, orient='records') for fn in slot_files)
slots['scum'] = slots['role'].str.contains("mafia|goon|wolf|serial.?killer", case=False) | slots['role'].str.contains(
    "SK")
slots['town'] = slots['role'].str.contains("town", case=False)
slots['sk'] = slots['role'].str.contains("serial.?killer", case=False) | slots['role'].str.contains("SK")

#kelvin's dirty hack (naughty)
slot_lookup = dict()

for s_id, s in slots.iterrows():
    for u in s.users:
        slot_lookup[(s.game_id, u)] = s

for game in tqdm(games):
    # create dummy column
    game['scum'] = False
    game['town'] = False
    game['sk'] = False
    game_id = game['game_id'].unique()[0]
    slots_game = slots[slots['game_id'] == game_id]

    for row_id, row in game.iterrows():
        slot = slot_lookup.get((game_id, row['author']))
        if slot is not None:
            row['scum'] = slot['scum']
            row['town'] = slot['town']
            row['sk'] = slot['sk']
