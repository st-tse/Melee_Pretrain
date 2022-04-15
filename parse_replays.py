from slippi import Game
from slippi.parse import parse
from slippi.parse import ParseEvent
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 
import argparse
from unicodedata import name

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
    return [game.frames[frame].ports[port].leader.pre, game.frames[frame].ports[port].leader.post]

def read_game(file_name):
    try:
        return Game(f'../Replays/{file_name}')
    except:
        print(f'Failed to parse {file_name}')

def add_replay(file_name, d, k):
    game = read_game(file_name)
    if game is None:
            print(f'{file_name} is None')
    else: 
        #get uses ports
        pls = [i for i, p in enumerate(game.start.players) if not(p is None)]
        player_info = game.start.players
        char_1 = player_info[pls[0]].character.name
        char_2 = player_info[pls[1]].character.name
        head = [file_name, char_1, char_2]
        #add each frame
        for f in range(len(game.frames)):
            d[k] = head + [f] + parse_frame_at_port(game, f, pls[0]) + parse_frame_at_port(game, f, pls[1])
            k += 1
    return d, k

def replays_to_df(replay_names):
    data = dict()
    k = 0
    for replay in replay_names:
        data, k = add_replay(replay, data, k)
    df = pd.DataFrame.from_dict(data, columns=['Game_ID','CHAR_P1', 'CHAR_P2','Frame', 'Pre_P1', 'Post_P1', 'Pre_P2', 'Post_P2'], orient='index')
    return df

def get_buttons(df):
    for i in [1,2]:
        pre = df[f'Pre_P{i}']
        df[f'B_damage_P{i}'] = pre.apply(lambda x : x.damage).values
        df[f'B_direction_P{i}'] = pre.apply(lambda x : x.damage).values
        df[f'B_joystick_x_P{i}'] = pre.apply(lambda x : x.joystick.x).values
        df[f'B_joystick_y_P{i}'] = pre.apply(lambda x : x.joystick.y).values
        df[f'B_position_x_P{i}'] = pre.apply(lambda x : x.position.x).values
        df[f'B_position_y_P{i}'] = pre.apply(lambda x : x.position.y).values
        df[f'B_cstick_x_P{i}'] = pre.apply(lambda x : x.cstick.x).values
        df[f'B_cstick_y_P{i}'] = pre.apply(lambda x : x.cstick.y).values
        df[f'B_state_P{i}'] = pre.apply(lambda x : x.state).values
        df[f'B_raw_analog_P{i}'] = pre.apply(lambda x : x.raw_analog_x).values
        df[f'B_buttons_physical_P{i}'] = pre.apply(lambda x : x.buttons.physical.value).values
        df[f'B_buttons_physical_P{i}'] = pre.apply(lambda x : x.buttons.logical.value).values
        df[f'B_triggers_physical_l_P{i}'] = pre.apply(lambda x : x.triggers.physical.l).values
        df[f'B_triggers_physical_r_P{i}'] = pre.apply(lambda x : x.triggers.physical.r).values
        df[f'B_triggers_logical_P{i}'] = pre.apply(lambda x : x.triggers.logical).values
    
    df = df.drop(['Pre_P1', 'Pre_P2'], axis = 1)
    return df

def df_column_switch(df, column1, column2):
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

df = get_states(df)
df = get_buttons(df)


