import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By  
import time
import urllib
import warnings
import argparse

#issues with getting the newer version of the function working
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser()
parser.add_argument('-p','--page', type=int, required = False, default = 1,
                    help='tournament page to start on, starts at 0, >= 1')
parser.add_argument('-t','--tour', type=int, required = False, default = 1,
                    help='tournament index to start on, starts at 0, >= 1')

args = parser.parse_args()

#check args
assert args.page >= 1
assert args.tour >= 1
args.page -= 1
args.tour -= 1

def download_slp(url_loc, dl = True):
    """
    download and slp file from a link
    set dl to false to only print names, used for testing
    """
    with urllib.request.urlopen(url_loc) as url:
        name = url_loc[url_loc.find('_') + 1:]
        if dl == True:
            data = open("Data/" + name, 'wb')
            data.write(url.read())
            data.close()
        else:
            print(name)

def page_download(soup):
    """
    download all .slp files from a page
    """
    empty = True
    for i, link in enumerate(soup.findAll('a')):
        file_url = link.get('href')
        if not (file_url is None) and (file_url[:30] == 'https://storage.googleapis.com'):
            download_slp(file_url.replace(' ','%20'), True) #toggle saving
            if empty:
                empty = False
    return empty

def scrape_tournament(driver):
    """
    scrape through a tournament
    """
    counter = 1
    empty = False
    while not empty:
        html = driver.page_source
        soup = bs(html)
        empty = page_download(soup)
        if not empty:
            try:
                next_page()
                counter += 1
            except:
                empty = True
                
        else:
            print(f'EMPTY PAGE FOUND ON {counter}')
    return counter

def next_page():
    ui_buttons = driver.find_elements_by_class_name('MuiSvgIcon-root')
    ui_buttons[-1].click()
    time.sleep(8)

#open window and go to the first page
driver = webdriver.Firefox()
url = 'https://slippi.gg/' 
driver.get(url)

#open menu to show links
button_ui = driver.find_element_by_xpath('/html/body/div[1]/div/header/div/header/div/button')
button_ui.click()

#click button for tournaments
button_tournament = driver.find_element_by_xpath('/html/body/div[3]/div[3]/div/ul/div[2]')
button_tournament.click()

time.sleep(5)

#skip to indented page
for _ in range(0,args.page):
    try:
        next_page()
    except:
        #write proper exception later
        assert 0==1

main_empty = False

while not main_empty:
    #check number of tournaments on the page
    tournament_buttons = driver.find_elements_by_partial_link_text('Browse all games')
    tournament_count = len(tournament_buttons)
    print(tournament_count)

    if tournament_count != 0:
        #change starting tournament
        if args.tour >= tournament_count:
            #write proper exception later
            assert 0==1

        for t in range(args.tour, tournament_count):
            #enter page
            print(f'TOURNAMENT {t + 1}')

            #skip to inputted tournament
            if args.tour > 0:
                t = args.tour

            #need to refetch links each time
            tournament_buttons = driver.find_elements_by_partial_link_text('Browse all games')
            tournament = tournament_buttons[t]

            #reset tournament counter
            args.tour = 0

            #scrape
            tournament.click()
            time.sleep(3)
            back_counter = scrape_tournament(driver)

            print(f'RETURNING ON PAGE: {back_counter}')

            # back_counter = 1
            #return to main tournament page
            for _ in range(back_counter):
                driver.back()
                time.sleep(1)

        print('NEXT TOURNAMENT PAGE')
        try:
            next_page()
        except:
            main_empty = True
        
print('Done')