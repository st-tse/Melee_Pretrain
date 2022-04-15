from slippi import Game
import pandas as pd
from os import listdir
from os import listdir, chdir
import argparse
import numpy as np

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, required = False, default = 32,
                    help='number of files to parse per batch')
parser.add_argument('-sb', '--start_batch', type=int, required=False, default = 1,
                    help='batch number to resume from')
parser.add_argument('-eb', '--end_batch', type=int, required=False,
                    help='batch number to resume from')
parser.add_argument('-r','--replays', type=str, default = 'Replays',
                    help='replays folder file path')
parser.add_argument('-o','--output', type=str, default = 'Data',
                    help='output folder file path')
parser.add_argument('-n','--out_name', type=str, default = 'replay_by_frame',
                    help='output file name') 
parser.add_argument('-off','--offset', type=str, default = 6,
                    help='frame offset between state and buttons to predict') 

args = parser.parse_args()

def parse_frame_at_port(game, frame, port):
    """
    gets player data from a Game object from port # 
    """
    return [game.frames[frame].ports[port].leader.pre, game.frames[frame].ports[port].leader.post]

def read_game(file_name):
    """
    read a game file using slippi parser returing Game object
    Prints out if fails to read
    """
    try:
        return Game(f'../{args.replays}/{file_name}')
    except:
        print(f'Failed to parse {file_name}')

def add_replay(file_name, d, k):
    """
    adds replay [file_name] to dictionary [d] and maintains an iterator [k] for the index numbers
    """
    game = read_game(file_name)
    if game is None:
            print(f'{file_name} is None')
    else: 
        #get used ports
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
    """
    add all replays in [replay_names] to a dataframe
    each frame is a row
    """
    data = dict()
    k = 0
    for replay in replay_names:
        data, k = add_replay(replay, data, k)
    df = pd.DataFrame.from_dict(data, columns=['Game_ID','CHAR_P1', 'CHAR_P2','Frame', 'Pre_P1', 'Post_P1', 'Pre_P2', 'Post_P2'], orient='index')
    return df

def get_states(df):
    """
    get states of the df
    """
    for i in [1,2]:
        post = df[f'Post_P{i}']
        df[f'S_airborne_P{i}'] = post.apply(lambda x : x.airborne).values
        df[f'S_damage_P{i}'] = post.apply(lambda x : x.damage).values
        df[f'S_direction_P{i}'] = post.apply(lambda x : x.direction).values
        df[f'S_hit_stun_P{i}'] = post.apply(lambda x : x.hit_stun).values
        df[f'S_position_x_P{i}'] = post.apply(lambda x : x.position.x).values
        df[f'S_position_y_P{i}'] = post.apply(lambda x : x.position.y).values
        df[f'S_shield_P{i}'] = post.apply(lambda x : x.shield).values
        df[f'S_state_P{i}'] = post.apply(lambda x : x.state).values
        df[f'S_state_age_P{i}'] = post.apply(lambda x : x.state_age).values
        df[f'S_stocks_P{i}'] = post.apply(lambda x : x.stocks).values
    
    df = df.drop(['Post_P1', 'Post_P2'], axis = 1)
    return df

def get_buttons(df):
    """
    get button inputs of the players
    """
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
    """
    swap names of 2 columns
    """
    i = list(df.columns)
    a, b = i.index(column1), i.index(column2)
    i[b], i[a] = i[a], i[b]
    df = df[i]
    return df

#col names for states and labels
X_cols = ['Game_ID', 'CHAR_P1', 'CHAR_P2', 'Frame', 'S_airborne_P1',
       'S_damage_P1', 'S_direction_P1', 'S_hit_stun_P1', 'S_position_x_P1',
       'S_position_y_P1', 'S_shield_P1', 'S_state_P1', 'S_state_age_P1',
       'S_stocks_P1', 'S_airborne_P2', 'S_damage_P2', 'S_direction_P2',
       'S_hit_stun_P2', 'S_position_x_P2', 'S_position_y_P2', 'S_shield_P2',
       'S_state_P2', 'S_state_age_P2', 'S_stocks_P2']

y_cols_P1 = ['Game_ID','Frame','B_damage_P1',
       'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
       'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
       'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
       'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
       'B_triggers_logical_P1']
          
y_cols_P2 =['Game_ID','Frame','B_damage_P2', 'B_direction_P2',
       'B_joystick_x_P2', 'B_joystick_y_P2', 'B_position_x_P2',
       'B_position_y_P2', 'B_cstick_x_P2', 'B_cstick_y_P2', 'B_state_P2',
       'B_raw_analog_P2', 'B_buttons_physical_P2', 'B_triggers_physical_l_P2',
       'B_triggers_physical_r_P2', 'B_triggers_logical_P2']

files = listdir(f'{args.replays}/')
chdir(f'{args.output}/')
first = True

batch_number = args.start_batch

if args.end_batch:
    end = min(args.end_batch * args.batch_size, len(files))
else:
    end = len(files)

#loop through batches to parse
for i in range((args.start_batch - 1) * args.batch_size, end, args.batch_size):
    
    df = replays_to_df(files[i:i+args.batch_size])

    df = get_states(df)
    df = get_buttons(df)

    #dataframes are states, P1 buttons, and P2 buttons
    df_X = df[X_cols]
    df_y_P1 = df[y_cols_P1]
    df_y_P2 = df[y_cols_P2]

    df_y_P1['Frame'] = df_y_P1['Frame'].apply(lambda x : x - args.offset)
    df_y_P2['Frame'] = df_y_P2['Frame'].apply(lambda x : x - args.offset)

    out1 = pd.merge(df_X, df_y_P1, on=['Game_ID', 'Frame'])

    #set P2 to be the same format as P1
    #always predicting "P2" buttons
    df_X2 = df_column_switch(df_X, 'CHAR_P1', 'CHAR_P2')

    df_y_P2 = df_y_P2.reindex(columns=['Game_ID','Frame','B_damage_P1',
           'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
           'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
           'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
           'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
           'B_triggers_logical_P1'])

    out2 = pd.merge(df_X2, df_y_P2, on=['Game_ID', 'Frame'])

    #combine dfs and save
    out = pd.concat([out1,out2],ignore_index=True)

    out.to_csv(f'{args.out_name}' + '.csv', index=False, mode = 'a', header = first)
    if first:
        first = False

    print(f'{batch_number}/{int(np.ceil(len(files) / args.batch_size))} Completed')
    batch_number += 1
