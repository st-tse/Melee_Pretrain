from unicodedata import name
from slippi import Game
from slippi.parse import parse
from slippi.parse import ParseEvent
import pandas as pd
from os import listdir, chdir
from os.path import isfile, join
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c','--count', type=int, required = False, default = -1,
                    help='number of files to parse')
parser.add_argument('-r','--replays', type=str, default = 'Replays',
                    help='replays folder')
parser.add_argument('-o','--output', type=str, default = 'Data',
                    help='output folder')
parser.add_argument('-n','--name', type=str, default = 'replay_by_frame',
                    help='output file name') 

args = parser.parse_args()

def parse_frame_at_port(game, frame, port):
    return [port, frame+1, game.frames[frame].ports[port].leader.pre, game.frames[frame].ports[port].leader.post]

def read_game(file_name, args):
    try:
        return Game(f'{args.replays}/{file_name}')
    except:
        print(f'Failed to parse {file_name}')

def add_replay(file_name, d, k, args):
    game = read_game(file_name, args)
    if game is None:
            print(f'{file_name} is None')
    else: 
        for i in range(len(game.start.players)):
            pl = game.start.players[i]
            if not(pl is None):
                char = pl.character.name
                head = [file_name, char]
                for f in range(len(game.frames)):
                    d[k] = head + parse_frame_at_port(game, f, i)
                    k += 1
    return d, k

def replays_to_df(replay_names, args):
    data = dict()
    k = 0
    for replay in tqdm(replay_names):
        data, k = add_replay(replay, data, k, args)
    df = pd.DataFrame.from_dict(data, columns=['Game_ID','CHAR', 'Port','Frame', 'Pre_frame', 'Post_frame'], orient='index')
    return df

def get_states(df):
    post = df['Post_frame']
    df['S_airborne'] = post.apply(lambda x : x.airborne).values
    df['S_damage'] = post.apply(lambda x : x.damage).values
    df['S_direction'] = post.apply(lambda x : x.direction).values
    df['S_hit_stun'] = post.apply(lambda x : x.hit_stun).values
    df['S_position_x'] = post.apply(lambda x : x.position.x).values
    df['S_position_y'] = post.apply(lambda x : x.position.y).values
    df['S_shield'] = post.apply(lambda x : x.shield).values
    df['S_state'] = post.apply(lambda x : x.state).values
    df['S_state_age'] = post.apply(lambda x : x.state_age).values
    df['S_stocks'] = post.apply(lambda x : x.stocks).values
    return df

def get_buttons(df):
    pre = df['Pre_frame']
    df['B_damage'] = pre.apply(lambda x : x.damage).values
    df['B_direction'] = pre.apply(lambda x : x.damage).values
    df['B_joystick_x'] = pre.apply(lambda x : x.joystick.x).values
    df['B_joystick_y'] = pre.apply(lambda x : x.joystick.y).values
    df['B_position_x'] = pre.apply(lambda x : x.position.x).values
    df['B_position_y'] = pre.apply(lambda x : x.position.y).values
    df['B_cstick_x'] = pre.apply(lambda x : x.cstick.x).values
    df['B_cstick_y'] = pre.apply(lambda x : x.cstick.y).values
    df['B_state'] = pre.apply(lambda x : x.state).values
    df['B_raw_analog'] = pre.apply(lambda x : x.raw_analog_x).values
    df['B_buttons_physical'] = pre.apply(lambda x : x.buttons.physical.value).values
    df['B_buttons_physical'] = pre.apply(lambda x : x.buttons.logical.value).values
    df['B_triggers_physical_l'] = pre.apply(lambda x : x.triggers.physical.l).values
    df['B_triggers_physical_r'] = pre.apply(lambda x : x.triggers.physical.r).values
    df['B_triggers_logical'] = pre.apply(lambda x : x.triggers.logical).values
    return df

files = listdir('Replays/')
df = replays_to_df(files[:args.count], args)
df = get_states(df)
df = get_buttons(df)

chdir(f'{args.output}/')
df.to_csv(f'{args.name}')

print('Done')