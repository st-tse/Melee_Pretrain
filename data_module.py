import torch
from torch.utils.data import DataLoader, Dataset

X_cols = ['CHAR_P1', 'CHAR_P2', 'S_airborne_P1',
   'S_damage_P1', 'S_direction_P1', 'S_hit_stun_P1', 'S_position_x_P1',
   'S_position_y_P1', 'S_shield_P1', 'S_state_P1', 'S_state_age_P1',
   'S_stocks_P1', 'S_airborne_P2', 'S_damage_P2', 'S_direction_P2',
   'S_hit_stun_P2', 'S_position_x_P2', 'S_position_y_P2', 'S_shield_P2',
   'S_state_P2', 'S_state_age_P2', 'S_stocks_P2']

y_cols = ['B_damage_P1',
   'B_direction_P1', 'B_joystick_x_P1', 'B_joystick_y_P1',
   'B_position_x_P1', 'B_position_y_P1', 'B_cstick_x_P1', 'B_cstick_y_P1',
   'B_state_P1', 'B_raw_analog_P1', 'B_buttons_physical_P1',
   'B_triggers_physical_l_P1', 'B_triggers_physical_r_P1',
   'B_triggers_logical_P1']

class FrameDataset(Dataset):
    def __init__(self,df):

        self.X = df[X_cols]
        self.y = df[y_cols]
        self.x_train=torch.tensor(self.X,dtype=torch.float32)
        self.y_train=torch.tensor(self.y,dtype=torch.float32)
        
    def __len__(self):
        return len(self.y_train)
  
    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

def split_data(df, test_size=0.2):
    games = list(df['Game_ID'].unique())
    count = len(games)
    i = int((1 - test_size) * count)
    train_ids, test_ids = games[:i], games[i:]
    train = df[df['Game_ID'].isin(train_ids)]
    test = df[df['Game_ID'].isin(test_ids)]
    train.drop('Game_ID', axis = 1, inplace=True)
    test.drop('Game_ID', axis = 1, inplace=True)
    return train, test

