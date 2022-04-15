from slippi import Game

def read_game(file_name):
    # try:
    #     return Game(f'{args.replays}/{file_name}')
    # except:
    #     print(f'Failed to parse {file_name}')
    return Game(f'Replays/{file_name}')

print(read_game('001AE992EB29_20210714T231243.slp'))

