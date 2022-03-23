import requests
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By  
import time
import urllib
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
                ui_buttons = driver.find_elements_by_class_name('MuiSvgIcon-root')
                ui_buttons[-1].click()
                time.sleep(8)
                counter += 1
            except:
                empty = True
                
        else:
            print(f'EMPTY PAGE FOUND ON {counter}')
    return counter

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

main_empty = False

while not main_empty:
    tournament_buttons = driver.find_elements_by_partial_link_text('Browse all games')
    tournament_count = len(tournament_buttons)
    print(tournament_count)

    if tournament_count != 0:
        for t in range(tournament_count):
            #enter page
            print(f'TOURNAMENT {t + 1}')
            tournament_buttons = driver.find_elements_by_partial_link_text('Browse all games')
            tournament = tournament_buttons[t]
            tournament.click()
            time.sleep(3)
            back_counter = scrape_tournament(driver)
            print(f'RETURNING ON PAGE: {back_counter}')
            # back_counter = 1
            #return to main
            for _ in range(back_counter):
                driver.back()
                time.sleep(1)

        print('NEXT TOURNAMENT PAGE')
        try:
            ui_buttons = driver.find_elements_by_class_name('MuiSvgIcon-root')
            ui_buttons[-1].click()
            time.sleep(5)
        except:
            main_empty = True
        
print('Done')