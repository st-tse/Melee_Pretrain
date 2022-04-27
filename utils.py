from slippi import Game
import pandas as pd
from os import listdir
from os import listdir, chdir
import argparse
import numpy as np

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
        if (char_1 in args.characters) and (char_2 in args.characters):
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