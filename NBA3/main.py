import requests
from bs4 import BeautifulSoup
from openpyxl import Workbook
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

url = 'https://basketball.realgm.com/nba/draft/past_drafts/'
player_base_url = 'https://basketball.realgm.com/'
#list for NCAA stats for drafts prior to 2019
records = []
#list for Rookie stats for drafts prior to 2019
rookie = []
##list for NCAA stats for 2019 draft
reco2019 = []

#List to determine if the player played in college
years = ['Fr *', 'So *', 'Jr *', 'Sr', 'Fr', 'So', 'Jr']
#List for NCAA stats columns
stats_column=['GP','GS','MIN','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT','OFF','DEF','TRB','AST','STL','BLK','PF','TOV','PTS']
#List to get specfic NBA rookie stats
stats_location = [3,17,18,19,20,23]


#function to scrape player data from url
def calc():
    #acquire data from 2003 to 2018 NBA draft
    for var in range(2003,2020):
        final_url = url+str(var)
        r = requests.get(final_url)
        #get the draft soup object
        soup = BeautifulSoup(r.text, 'html.parser')

        player_records = soup.find_all('tr')
        upto = 61
        skip = 31
        #For drafts prior to 2005, 1 less player drafted in the NBA
        if var<2005:
            upto = 60
            skip = 30
        #Get information for the all the NBA picks
        for a in range(1,upto):
            if a == skip:
                continue
            player = player_records[a];
            #Get the player info; E.g position
            player_info=player.find_all('td')
            year = player_info[10].text
            position = player_info[4].text
            pick = player_info[0].text

            #If the player played in a US college, get the NCAA stats
            #international player pages are inconsistent
            if year in years:
                calculate_stats(player,position,pick,'career per_game', 'NCAA DI', var)


#function to get NCAA and Rookie stats
def calculate_stats(player, position,pick, attrs, division, year):
    #Get the player page url
    player_final_url = player_base_url+player.find('a')['href']
    r2 = requests.get(player_final_url)
    #Get the player soup object
    soup2 = BeautifulSoup(r2.text, 'html.parser')

    player_entire = soup2.find_all('tr', attrs={'class': attrs})
    ro = soup2.find_all('tr', attrs={'class': 'per_game'})
    i = 0
    #Find the NCAA stats for each player out of the entire playing career
    while i<len(player_entire):
        player_points=player_entire[i].find_all('td')
        if len(player_points)>0:
            if player_points[1].text == division:
                break
        i+=1
    #If it exists, added the NCAA stats for this player to the list of records
    if i < len(player_entire):

        #Get the object for stats, miscellaneous stats and advance stats
        ncaa_record=player_entire[i].find_all('td')
        ncaa_record_misc = player_entire[i+2].find_all('td')
        ncaa_record_advance = player_entire[i+3].find_all('td')
        #Need to grab the rookie stats for players prior to 2019 NBA draft
        #2019 Draft players don't have rookie stats since they haven't played
        #in NBA
        if(year<2019):
            #Attempt to get rookie data as not all draft picks end up playing
            #in NBA
            canAdd = rookie_method(player,ro)
            #Extract necessary information from NCAA career
            if len(ncaa_record)>23 and canAdd:
                records.append(player.find('a').text)
                records.append(year)
                records.append(pick)
                records.append(position)
                for ind in range(3, 24):
                    records.append(ncaa_record[ind].text)


                records.append(ncaa_record_misc[len(ncaa_record_misc)-1].text)
                records.append(ncaa_record_advance[5].text)
                records.append(ncaa_record_advance[len(ncaa_record_advance)-1].text)
        #Extract only NCAA data for 2019 NBA Draft picks
        else:
            if len(ncaa_record)>23:
                reco2019.append(player.find('a').text)
                reco2019.append(year)
                reco2019.append(pick)
                reco2019.append(position)
                for ind in range(3, 24):
                    reco2019.append(ncaa_record[ind].text)

                reco2019.append(ncaa_record_misc[len(ncaa_record_misc)-1].text)
                reco2019.append(ncaa_record_advance[5].text)
                reco2019.append(ncaa_record_advance[len(ncaa_record_advance)-1].text)


#Function to get Rookie Stats
def rookie_method(player,ro):
    #Get rookie object
    rookie_stats = ro[1].find_all('td')
    #Used a try/except due to formatting of HTML, determine, if player played
    #in NBA, if not return False
    try:
        if 'Summer_League' in rookie_stats[1].find('a')['href']:
            return False
    except TypeError:
        return False
    #Get rookie stats
    if rookie_stats[2].text not in years:
        rookie.append(player.find('a').text)
        for ind in range(0, len(stats_location)):
            rookie.append(rookie_stats[stats_location[ind]-1].text)
        return True

    return False


#call the function
calc()
#Get numpy arrays and reshape it to make it easier to work with pandas framework
dummy_array = np.array(records)
ncaa_stats = np.reshape(dummy_array, (-1, 28))
dummy_array2 = np.array(rookie)
rook_stats = np.reshape(dummy_array2, (-1, 7))
dummy_array3 = np.array(reco2019)
ncaa_stats_2019 = np.reshape(dummy_array3, (-1, 28))

#Export all the data into a csv file
df = pd.DataFrame(ncaa_stats, columns=(['player name','year','pick','position']+stats_column+['WS','eFG%','PER']))
df.set_index('player name', inplace=True)
df.to_csv('toppick_records.csv')
df2 = pd.DataFrame(rook_stats, columns=['player name','GP','TRB','AST','STL','BLK','PTS'])
df2.set_index('player name', inplace=True)
df2.to_csv('rookie_records.csv')
df3 = pd.DataFrame(ncaa_stats_2019, columns=(['player name','year','pick','position']+stats_column+['WS','eFG%','PER']))
df3.set_index('player name', inplace=True)
df3.to_csv('toppick2019_records.csv')
