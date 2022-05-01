import torch
from torch.utils.data import DataLoader, Dataset

class FrameDataset(Dataset):
    def __init__(self,X,y):
        self.x_train=torch.tensor(X,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

X_cols_b = ['CHAR_P1', 'CHAR_P2', 'S_airborne_P1',
   'S_damage_P1', 'S_direction_P1', 'S_hit_stun_P1', 'S_position_x_P1',
   'S_position_y_P1', 'S_shield_P1', 'S_state_P1', 'S_state_age_P1',
   'S_stocks_P1', 'S_airborne_P2', 'S_damage_P2', 'S_direction_P2',
   'S_hit_stun_P2', 'S_position_x_P2', 'S_position_y_P2', 'S_shield_P2',
   'S_state_P2', 'S_state_age_P2', 'S_stocks_P2']

y_cols_b = ['B_damage_P1',
   'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
   'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
   'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
   'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
   'B_triggers_logical_P1']

X_cols_s = ['CHAR_P1', 'CHAR_P2', 'S_airborne_P1_x',
       'S_damage_P1_x', 'S_direction_P1_x', 'S_hit_stun_P1_x',
       'S_position_x_P1_x', 'S_position_y_P1_x', 'S_shield_P1_x',
       'S_state_P1_x', 'S_state_age_P1_x', 'S_stocks_P1_x', 'S_airborne_P2_x',
       'S_damage_P2_x', 'S_direction_P2_x', 'S_hit_stun_P2_x',
       'S_position_x_P2_x', 'S_position_y_P2_x', 'S_shield_P2_x',
       'S_state_P2_x', 'S_state_age_P2_x', 'S_stocks_P2_x', 'B_damage_P1',
       'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
       'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
       'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
       'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
       'B_triggers_logical_P1']

y_cols_s = ['S_airborne_P1_y', 'S_damage_P1_y',
       'S_direction_P1_y', 'S_hit_stun_P1_y', 'S_position_x_P1_y',
       'S_position_y_P1_y', 'S_shield_P1_y', 'S_state_P1_y',
       'S_state_age_P1_y', 'S_stocks_P1_y', 'S_airborne_P2_y', 'S_damage_P2_y',
       'S_direction_P2_y', 'S_hit_stun_P2_y', 'S_position_x_P2_y',
       'S_position_y_P2_y', 'S_shield_P2_y', 'S_state_P2_y',
       'S_state_age_P2_y', 'S_stocks_P2_y']

def split_data(df, test_size=0.2, dataset_type='b'):
    games = list(df['Game_ID'].unique())
    count = len(games)
    i = int((1 - test_size) * count)
    train_ids, test_ids = games[:i], games[i:]
    train = df[df['Game_ID'].isin(train_ids)]
    test = df[df['Game_ID'].isin(test_ids)]
    train.drop('Game_ID', axis = 1, inplace=True)
    test.drop('Game_ID', axis = 1, inplace=True)

    if dataset_type == 'b':
        X_cols = X_cols_b
        y_cols = y_cols_b
    else:
        X_cols = X_cols_s
        y_cols = y_cols_s
        
    x_train = train[X_cols]
    y_train = train[y_cols]
    x_test = test[X_cols]
    y_test = test[y_cols]

    return x_train, y_test, x_test, y_test

