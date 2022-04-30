from slippi import Game
import pandas as pd
from os import listdir
from os import listdir, chdir
import argparse
import numpy as np

from utils import *

import warnings
from pandas.core.common import SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

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
parser.add_argument('-off','--offset', type=int, default = 6,
                    help='frame offset between state and buttons to predict')
parser.add_argument('-c', '--characters', type=list,  nargs='+', required = False, default = ['FOX', 'MARTH'],
                    help='characters to use, otherwise skip. selects all pairwise combinations in the list including mirror')


args = parser.parse_args()

mapping = {'B_damage_P2': 'B_damage_P1',
 'B_direction_P2': 'B_direction_P1',
 'B_joystick_x_P2': 'B_joystick_x_P1',
 'B_joystick_y_P2': 'B_joystick_y_P1',
 'B_position_x_P2': 'B_position_x_P1',
 'B_position_y_P2': 'B_position_y_P1',
 'B_cstick_x_P2': 'B_cstick_x_P1',
 'B_cstick_y_P2': 'B_cstick_y_P1',
 'B_state_P2': 'B_state_P1',
 'B_raw_analog_P2': 'B_raw_analog_P1',
 'B_buttons_physical_P2': 'B_buttons_physical_P1',
 'B_triggers_physical_l_P2': 'B_triggers_physical_l_P1',
 'B_triggers_physical_r_P2': 'B_triggers_physical_r_P1',
 'B_triggers_logical_P2': 'B_triggers_logical_P1'}

#col names for states and labels
X_cols = ['Game_ID', 'CHAR_P1', 'CHAR_P2', 'Frame', 'S_airborne_P1',
       'S_damage_P1', 'S_direction_P1', 'S_hit_stun_P1', 'S_position_x_P1',
       'S_position_y_P1', 'S_shield_P1', 'S_state_P1', 'S_state_age_P1',
       'S_stocks_P1', 'S_airborne_P2', 'S_damage_P2', 'S_direction_P2',
       'S_hit_stun_P2', 'S_position_x_P2', 'S_position_y_P2', 'S_shield_P2',
       'S_state_P2', 'S_state_age_P2', 'S_stocks_P2']

x_cols_P1 = ['B_damage_P1',
       'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
       'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
       'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
       'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
       'B_triggers_logical_P1']
          
x_cols_P2 =['B_damage_P2', 'B_direction_P2',
       'B_joystick_x_P2', 'B_joystick_y_P2', 'B_position_x_P2',
       'B_position_y_P2', 'B_cstick_x_P2', 'B_cstick_y_P2', 'B_state_P2',
       'B_raw_analog_P2', 'B_buttons_physical_P2', 'B_triggers_physical_l_P2',
       'B_triggers_physical_r_P2', 'B_triggers_logical_P2']

y_cols = ['Game_ID','Frame', 'S_airborne_P1',
       'S_damage_P1', 'S_direction_P1', 'S_hit_stun_P1', 'S_position_x_P1',
       'S_position_y_P1', 'S_shield_P1', 'S_state_P1', 'S_state_age_P1',
       'S_stocks_P1', 'S_airborne_P2', 'S_damage_P2', 'S_direction_P2',
       'S_hit_stun_P2', 'S_position_x_P2', 'S_position_y_P2', 'S_shield_P2',
       'S_state_P2', 'S_state_age_P2', 'S_stocks_P2']

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
    
    df = replays_to_df(files[i:i+args.batch_size], args)

    df = get_states(df)
    df = get_buttons(df)

    #dataframes are states, P1 buttons, and P2 buttons
    df_y_target = df[y_cols]
    df_y_target['Frame'] = df_y_target['Frame'].apply(lambda x : x - args.offset)

    df_X_P1 = df[X_cols + x_cols_P1]
    df_X_P2 = df[X_cols + x_cols_P2]

    df_X_P2 = df_column_switch(df_X_P2, 'CHAR_P1', 'CHAR_P2')
    df_X_P2.rename(columns=mapping, inplace=True)

    out1 = pd.merge(df_X_P1, df_y_target, on=['Game_ID', 'Frame'])
    out2 = pd.merge(df_X_P2, df_y_target, on=['Game_ID', 'Frame'])

    out = pd.concat([out1,out2],ignore_index=True)

    out.to_csv(f'{args.out_name}' + '.csv', index=False, mode = 'a', header = first)
    if first:
        first = False

    print(f'{batch_number}/{int(np.ceil(len(files) / args.batch_size))} Completed')
    batch_number += 1
