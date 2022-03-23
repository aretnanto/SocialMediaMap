import json
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import DBSCAN

####Will be using folium to generate map information

##SPOTIFY -- GET TIMES for listening to music
###TO Do Make Big Json for Spotify -- Do so before running
spotify_data0 = pd.read_json('data/StreamingHistory0.json')
spotify_data1 = pd.read_json('data/StreamingHistory1.json')
spotify_data2 = pd.read_json('data/StreamingHistory2.json')
spotify_data3 = pd.read_json('data/StreamingHistory3.json')
spotify_data4 = pd.read_json('data/StreamingHistory4.json')
merged_spotify = pd.concat([spotify_data0,spotify_data1,spotify_data2,spotify_data3,spotify_data4]) 
merged_spotify = merged_spotify.drop(columns = ['artistName','trackName'])
merged_spotify = merged_spotify.drop_duplicates(subset=['endTime'])
merged_spotify['endTime'] = pd.to_datetime(merged_spotify['endTime'],utc =True)
merged_spotify['Time'] = merged_spotify['endTime'] ##if within the duration then it is true
merged_spotify = merged_spotify.set_index('Time')
merged_spotify = merged_spotify.rename(columns={"endTime": "SpotEndTime","msPlayed":"SpotDur"})


##NETFLIX -- Watching Activity//Assumption only one account
netflix_data = pd.read_csv('data/NetflixViewingActivity.csv')
#Assume that each profile is the same person
netflix_data = netflix_data.drop(columns = ['Profile Name','Attributes','Latest Bookmark','Bookmark','Title','Country','Supplemental Video Type', 'Device Type'])
netflix_data['Start Time'] = pd.to_datetime(netflix_data['Start Time'],utc =True)
netflix_data['Time'] = netflix_data['Start Time'] ##variable we will use to create sampling
epoch = dt.utcfromtimestamp(0)
netflix_data['Duration'] = pd.to_timedelta(netflix_data['Duration'])
netflix_data = netflix_data.set_index('Time')
netflix_data = netflix_data.rename(columns={"Start Time": "NetStartTime","Duration": "NetDur"})
#if within duration window set as true

##SNAPCHAT -- Location History
#snapchat_data = pd.read_json('data/Snapchat_Location.json')
snapchat_data = json.load(open('data/Snapchat_Location.json'))
snapchat_data = snapchat_data['Location History'] #Variable I'm interested in
snapchat_data = pd.DataFrame.from_dict(snapchat_data, orient='columns')
snapchat_data[['Latitude','Longitude']] = snapchat_data['Latitude, Longitude'].str.split(", ",expand=True)
##Assumption the += will not matter in the end
snapchat_data['Longitude'] = snapchat_data['Longitude'].str.split(' ').str[0].astype('float')
snapchat_data['Latitude'] = snapchat_data['Latitude'].str.split(' ').str[0].astype('float')
snapchat_data = snapchat_data.drop(columns='Latitude, Longitude')
snapchat_data = snapchat_data.drop_duplicates(subset=['Time'])
snapchat_data['Time'] = pd.to_datetime(snapchat_data['Time'],utc =True)
snapchat_data = snapchat_data.set_index('Time')

#only want date time range with snapchat data
#mintime = snapchat_data.index.min()
maxtime = snapchat_data.index.max()
mintime = snapchat_data.index.min()

##filter
merged_spotify = merged_spotify.loc[(merged_spotify.index >= mintime) & (merged_spotify.index <= maxtime)]
netflix_data = netflix_data.loc[(netflix_data.index >= mintime) & (netflix_data.index <= maxtime)]

sample_rate = '10t'
netflix_data = netflix_data.resample(sample_rate).bfill()
merged_spotify = merged_spotify.resample(sample_rate).bfill()
snapchat_data = snapchat_data.resample(sample_rate).ffill()
user_profile = pd.concat([netflix_data,merged_spotify,snapchat_data],axis=1,sort=False).reset_index()
user_profile = user_profile.dropna(subset=['Longitude']).reset_index()


###Logic to Deal With Activity
###Define what activity is. For spotify and netflix we have the run times. For youtube/tiktok no also figure out how to pad
##Spotify Activity
SpotifyActivity = (user_profile['SpotEndTime'] - user_profile['Time'])
spotmask = (SpotifyActivity.dt.total_seconds() * 1000) >= user_profile['SpotDur']
user_profile['Spotify_Activity'] = spotmask

#Netflix Actity 
NetflixActivity = (user_profile['NetStartTime'] - user_profile['Time'])
netmask = (NetflixActivity.dt.total_seconds()) <= user_profile['NetDur'].dt.total_seconds()
user_profile['Netflix_Activity'] = netmask

##ML-Spatial Focus Viz 
###Concern is assumption that cluster density is the same
train_data = user_profile[['Longitude','Latitude']]
out_labels = DBSCAN(eps=.005).fit_predict(train_data)
bins, counts = np.unique(out_labels, return_counts=True)
large_clusters = np.count_nonzero(np.where(counts >= len(user_profile)/len(bins))) ###Assuming Uniform Distribition - Not True
top_idx = np.argsort(counts)[-large_clusters:] -1 ##have some sort of definition that is considered significant. #adding one because -1

#Maybe have stationary/area of significance

reduced_label = []
for i in range(0,len(out_labels)):
    if out_labels[i] in top_idx:
        idx = np.where(top_idx == out_labels[i])[0][0]
        reduced_label.append(idx)
    else:
        reduced_label.append(-1)
user_profile['reduced_label'] = reduced_label

plt.scatter(user_profile['Longitude'],user_profile['Latitude'], c= reduced_label,s=50)
plt.axis('off')
plt.show()

##Map Viz
##Frequent Visit LISTENING TO MUSIC, AWAY LISTENING TO MUSIC, Frequent WATCHING, AWAY WATCHING
user_profile['Location'] = user_profile['reduced_label'].apply(lambda x: "Frequent Location" if x != -1 else "Away")
user_profile['Activity'] = user_profile['reduced_label'].apply(lambda x: "Frequent Location" if x != -1 else "Away")
user_profile['Spotify_Activity_Str'] = user_profile['Spotify_Activity'].apply(lambda x: "Yes" if x  else "No")
user_profile['Netflix_Activity_Str'] = user_profile['Netflix_Activity'].apply(lambda x: "Yes" if x  else "No")
m = folium.Map(location=[user_profile['Latitude'].median(), user_profile['Longitude'].median()],zoom_start=4)
for i in range(0,len(user_profile)):
    if user_profile['Location'][i] == 'Frequent Location' and ((user_profile['Spotify_Activity'][i] or user_profile['Netflix_Activity'][i])):
        folium.Circle(
        [user_profile['Latitude'].loc[i],user_profile['Longitude'].loc[i]],
        radius = 10,
        popup='Let''s Advertise',
        color="crimson",
        fill = True
        ).add_to(m)
    elif user_profile['Location'][i] == 'Frequent Location' and not ((user_profile['Spotify_Activity'][i] or user_profile['Netflix_Activity'][i])):
        folium.Circle(
        [user_profile['Latitude'].loc[i],user_profile['Longitude'].loc[i]],
        radius = 10,
        opacity=0.8,
        popup='User Not Paying Attention',
        color="green",
        ).add_to(m)
    else:
        folium.Circle(
        [user_profile['Latitude'].loc[i],user_profile['Longitude'].loc[i]],
        radius = 10,
        opacity=0.8,
        popup=user_profile['Location'].loc[i] + '\n\nAudio Engagement: ' + user_profile['Spotify_Activity_Str'].loc[i] + '\nAudiovisual Engagement: ' + user_profile['Netflix_Activity_Str'].loc[i],
        ).add_to(m)
m.save('data/personal_tracking.html')
