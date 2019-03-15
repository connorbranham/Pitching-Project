"""
Connor Branham
This is my first entirely original data science project.
The goal of this project is to answer basic questions about pitching and hitting choices.


"""
import plotly
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
import operator

os.chdir('C:\\Users\\conno\\Downloads\\PythonProj\\PythonData')


bat=pd.read_csv('atbats.csv')
games=pd.read_csv('games.csv')
pitch=pd.read_csv('pitches.csv', nrows=40_000)
player=pd.read_csv('player_names.csv')

#The Pitch data frame does not specify the result of the at bat nor does it give the at bat context. Combining these will help later.
pitchbat=pd.merge(bat, pitch, on='ab_id')

#Dropping null values. In 40,000 observations there are only 122 missing.
nullcol=pitch.columns[pitch.isnull().any()]
pitch[nullcol].isnull().sum()
pitch.dropna()

#Plots pitch location. This is a rather blunt plot, but it does give us some perspective on the variables px and py, which will be valuable later.
pl1=sns.scatterplot(x=pitch.px, y=pitch.pz, data=pitch)
pl2=sns.scatterplot(x=pitch.px, y=pitch.pz, data=pitch, hue=pitch.code)


# Creating a data set that consists purely of umpire calls. This shows the effective strike zone, since all of this pitches were called by the umpire.
BS= pitch[(pitch['code']=='C') | (pitch['code']=='B')]
pl3=sns.scatterplot(x=BS.px, y=BS.pz, hue=BS.code, data=BS, alpha=0.1)
pl3

#This creates a dataframe that only counts pitches that are considered part of the Data sets "strike zone."
BSzone= BS[(BS['zone']>=1) & (BS['zone']<=9)]

# This plot shows us the frequency of the pitches if we break the strike zone into 36 squares.
pl4=plt.hist2d(BSzone['px'], BSzone['pz'], bins=(6,6), cmap=plt.cm.Reds)
plt.show()

# This data frame consist of only called strikes in the strike zone.
strikes= pitchbat[(pitchbat['code']=='C') & ((pitchbat['zone']>=1) & (pitchbat['zone']<=9)) ]

#Try to find the percenatge of zone 1,2, and 3 that are called strikes
BS['strike']=0
BS['strike'][BS['code']=='C']=1
BS[(BS['pz']>=3.5)&(BS['pz']<=4)&(BS['px']>=-0.5)&(BS['px']<=0.5)]['strike'].describe()

#Alternates between Right and Left pichers
dbat=['R','L']

# This graphs will show us where the most strikes are thrown depending on the dominant hand of the batter or pitcher.
for key in dbat:
    for key2 in dbat:
        plt.subplot
        strikezone=plt.hist2d(strikes[((strikes['p_throws']==key)&(strikes['stand']==key2))]['px'], strikes[((strikes['p_throws']==key)&(strikes['stand']==key2))]['pz'], bins=(3,3), cmap=plt.cm.Reds)
        plt.title('Pitcher : ' + key + ' Batter : ' + key2, fontsize=30)
        plt.show()

# This graph shows the difference in pitching location depending on the pitch the pitcher throws.
dpitch=['FF', 'FT', 'CH', 'CU', 'SL']
for key in dpitch:
    strikezone=plt.hist2d(strikes[strikes['pitch_type']==key]['px'], strikes[strikes['pitch_type']==key]['pz'], bins=(3,3), cmap=plt.cm.Reds)
    plt.title(key)
    plt.show()

#Hits by strike zone
pitchbat['Hit']=0
pitchbat.loc[(((pitchbat['code']=='X') | (pitchbat['code']=='E') | (pitchbat['code']=='D')) & ((pitchbat['event']=='Single') | (pitchbat['event']=='Double') | (pitchbat['event']=='Triple') | (pitchbat['event']=='Home Run'))), 'Hit']=1



# If the batter swings at a pitch in the zone, we want to know the frequency of Hit, Out, and Foul percentage by zone.
#This data frame consist of only pitches that were swung at.
swing= pitchbat[((pitchbat['code']=='F') | (pitchbat['code']=='S')| (pitchbat['code']=='X')| (pitchbat['code']=='D')| (pitchbat['code']=='E')| (pitchbat['code']=='T')| (pitchbat['code']=='L')| (pitchbat['code']=='W'))]
swing['Hit']=0
swing['Hit'][(((pitchbat['event']=='Single') | (pitchbat['event']=='Double') | (pitchbat['event']=='Triple') | (pitchbat['event']=='Home Run')) & ((swing['code']=='X')|(swing['code']=='D')|(swing['code']=='E')))]=1
swing['HR']=0
swing['HR'][((swing['event']=='Home Run') & ((swing['code']=='X')|(swing['code']=='D')|(swing['code']=='E')))]=1
swing['Foul']=0
swing['Foul'][((swing['code']=='F')|(swing['code']=='T'))]=1
swing['Out']=0
swing['Out'][(((swing['code']=='X')|(swing['code']=='D')|(swing['code']=='E')) & ((swing['event']=='Flyout')|(swing['event']=='Groundout')|(swing['event']=='Pop Out')|(swing['event']=='Lineout')|(swing['event']=='Forceout')))]=1

# These Matrices will hold the hit percentage, HR percentage, Foul percentage, out percentage, hit to out ratio, and HR to out ratio by zone in the strike zone.
hitmat= np.zeros((8,3,3))
HRmat= np.zeros((8,3,3))
foulmat=np.zeros((8,3,3))
outmat=np.zeros((8,3,3))
hitoutmat=np.zeros((8,3,3))
HRoutmat=np.zeros((8,3,3))
d={'':0 , 'R': 1, 'L': 2, 'FF': 3 , 'FT': 4, 'CU': 5, 'CH': 6, 'SL': 7}

batterd={'R': 1, 'L': 2}
pitchd={'FF': 3 , 'FT': 4, 'CU': 5, 'CH': 6, 'SL': 7}

for i in [0,1,2]:
    for n in [0, 1, 2]:
        hitmat[0, i, n]= swing['Hit'][swing['zone']== (i+1)*(n+1)].mean()
        HRmat[0, i, n]= swing['HR'][swing['zone']== (i+1)*(n+1)].mean()
        foulmat[0, i, n]=swing['Foul'][swing['zone']== (i+1)*(n+1)].mean()
        outmat[0, i, n]=swing['Out'][swing['zone']== (i+1)*(n+1)].mean()
        hitoutmat[0, i, n]= (swing['Hit'][swing['zone']== (i+1)*(n+1)].mean())/(swing['Out'][swing['zone']== (i+1)*(n+1)].mean())
        HRoutmat[0, i, n]= (swing['HR'][swing['zone']== (i+1)*(n+1)].mean())/(swing['Out'][swing['zone']== (i+1)*(n+1)].mean())

for key, value in batterd.items():
    for i in [0,1,2]:
        for n in [0, 1, 2]:
            hitmat[value, i, n]= swing['Hit'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean()
            HRmat[value, i, n]= swing['HR'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean()
            foulmat[value, i, n]=swing['Foul'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean()
            outmat[value, i, n]=swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean()
            hitoutmat[value, i, n]= (swing['Hit'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean())/(swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean())
            HRoutmat[value, i, n]= (swing['HR'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean())/(swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['stand']==key))].mean())
            
            
for key, value in pitchd.items():
    for i in [0,1,2]:
        for n in [0, 1, 2]:
            hitmat[value, i, n]= swing['Hit'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean()
            HRmat[value, i, n]= swing['HR'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean()
            foulmat[value, i, n]=swing['Foul'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean()
            outmat[value, i, n]=swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean()
            hitoutmat[value, i, n]= (swing['Hit'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean())/(swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean())
            HRoutmat[value, i, n]= (swing['HR'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean())/(swing['Out'][((swing['zone']== (i+1)*(n+1)) & (swing['pitch_type']==key))].mean())
matd={'hitmat':hitmat, 'HRmat':HRmat, 'foulmat':foulmat, 'outmat':outmat, 'hitoutmat':hitoutmat, 'HRoutmat':HRoutmat}          

os.chdir('C:\\Users\\conno\\Downloads\\PythonProj\\PythonData\\Graphs')
for matname, mat in matd.items():
    for key, value in d.items():
        plt.clf()
        plt.imshow(mat[value,:,:])
        plt.title(matname+key, fontsize=30)
        plt.ylabel('Vertical Placement', fontsize=20)
        plt.xlabel('Horizontal Placement', fontsize=20)
        plt.colorbar()
        plt.savefig(matname+key+'.png')


os.chdir('C:\\Users\\conno\\Downloads\\PythonProj\\PythonData')


BS= BS[['break_angle', 'break_length','end_speed', 'px', 'pz','code']]
BS=BS.dropna()
# In Preparation for using KNN, We standardize the variables.
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(BS.drop('code', axis=1))
scaled=scaler.transform(BS.drop('code', axis=1))
kBS=pd.DataFrame(scaled, columns=BS.columns[:-1])


# Split the data.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(scaled, BS['code'], test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
#
from sklearn.metrics import classification_report, confusion_matrix

#Printing the results.
print(confusion_matrix(y_test, pred))

print(classification_report(y_test,pred))
#We find that KNN can classify balls with 91 percent accuracy and strikes with 86 percent accuracy.

#We will change the data set to see if the other variables like break affected the umpires calls. Now we will do it using only coordinates, which should be the basis of balls and strikes.
BS=BS[['px','pz','code']]
scaler.fit(BS.drop('code', axis=1))
scaled2=scaler.transform(BS.drop('code', axis=1))
kBS3=pd.DataFrame(scaled2, columns=BS.columns[:-1])

X_train2, X_test2, y_train2, y_test2=train_test_split(scaled2, BS['code'], test_size=0.3)
knn2=KNeighborsClassifier(n_neighbors=10)
knn2.fit(X_train2, y_train2)
pred2 = knn2.predict(X_test2)

print(confusion_matrix(y_test2, pred2))
print(classification_report(y_test2,pred2))
# It preforms slightly better. Ballscan be predicted with 92 percent accuracy and Strikes again with 86 percent accuracy.

# Now compare to SVM classification that would theoretically recreate the strike zone

from sklearn.svm import SVC
model = SVC()
model.fit(X_train2,y_train2)
predictions = model.predict(X_test2)
print(confusion_matrix(y_test2,predictions))
print(classification_report(y_test2,predictions))
# This works with a simlar degree of accuracy to KNN. 93 percent for called balls and 86 percent for called strikes.

#Use Grid Search 

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
# This works with a simlar degree of accuracy to SVM. 92 percent for called balls and 87 percent for called strikes.

#Taking normal pitches : FF 4 seam FB, CH Changeup, CU Curveball, FT Two-seam Fastball, Sl Slider 
#And taking the most common results : B called Ball, C called strike, F foul, X hit but out, D hit, E hit with RBI, H hit by pitch
pitch1=pitch[((pitch['pitch_type']=='FF') | (pitch['pitch_type']=='CU')| (pitch['pitch_type']=='CH')| (pitch['pitch_type']=='FT')| (pitch['pitch_type']=='SL'))&((pitch['code']=='B')|(pitch['code']=='C')|(pitch['code']=='F')|(pitch['code']=='X')|(pitch['code']=='D')|(pitch['code']=='E')|(pitch['code']=='B'))]

# Check the count of pitch type by balls and strikes.
pitch1.dropna()
pitchBS=pitch1[(pitch1['code']=='B') | (pitch1['code']=='T') |(pitch1['code']=='C') |(pitch1['code']=='S') |(pitch1['code']=='F')]
pitch_counts_S=pitchBS[(pitchBS['code']=='C') | (pitchBS['code']=='S') | (pitchBS['code']=='F') | (pitchBS['code']=='FT')]['pitch_type'].value_counts()
pitch_counts_B=pitchBS[pitchBS['code']=='B']['pitch_type'].value_counts()
pitch_counts_S
pitch_counts_B

plt.clf()
R=np.arange(5)
p1=plt.bar(R, pitch_counts_S)
p2=plt.bar(R, pitch_counts_B, bottom=pitch_counts_S)
p1
plt.ylabel('Pitch Count')
plt.xlabel('Pitch Type')
plt.title('Balls and Strikes by Pitch')
plt.xticks(R, ('FF','SL','FT','CH','CU'))
plt.legend((p1[0], p2[0]),('Strike', 'Ball'))
plt.show()
# There does not appear to be any pitch that is more likely to be a ball than the others.

# Prepping the pitch data.
pitch1=pitch1[['pitch_type','b_count','break_angle','break_length','end_speed','outs','pitch_num','px','pz','s_count','spin_dir','spin_rate','start_speed']]

pitch1['change_speed']=pitch1['start_speed']-pitch1['end_speed']
pitch2=pitch1
pitch2['pitch_type']=pitch2['pitch_type'].astype('category')
pitch2=pitch2[['pitch_type','change_speed','b_count','break_angle','break_length','end_speed','outs','pitch_num','px','pz','s_count','spin_dir','spin_rate','start_speed']]

pitch3=pd.get_dummies(pitch2['pitch_type'])
pitch2=pitch2[['change_speed','b_count','break_angle','break_length','end_speed','outs','pitch_num','px','pz','s_count','spin_dir','spin_rate','start_speed']]

pitch3['FF'].describe()
pitch1=pd.concat([pitch2,pitch3], axis=1)

pit=['FF','CH','CU','FT','SL']
for i in pit:
    pitchloop=pitch1[[i,'b_count', 'change_speed','break_angle','break_length','end_speed','outs','pitch_num','px','pz','s_count','spin_dir','spin_rate','start_speed']]

    scaler=StandardScaler()
    scaler.fit(pitchloop.drop(i, axis=1))
    scaled=scaler.transform(pitchloop.drop(i, axis=1))

    indep=pd.DataFrame(scaled, columns=pitchloop.columns[1:])
    
    X_train, X_test, y_train, y_test=train_test_split(scaled, pitchloop[i], test_size=0.3)

    knn=KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    print(i, 'results in:')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test,pred))

#Using KNN we can predict whether it is a: Slider with 86% accuracy, Two Seam FB with 78% accuracy, Curveball with 89% accuracy, Changeup with 89% accuracy, and four seam fastball with 86% accuracy.
    

#We will use decision tree classification to see if we can predict a pitchers pitch just using the game situation.
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier

dtree=DecisionTreeClassifier()
#rfc = RandomForestClassifier(n_estimators=600)

pitch1=pitch[(((pitch['pitch_type']=='FF') | (pitch['pitch_type']=='CU')| (pitch['pitch_type']=='CH')| (pitch['pitch_type']=='FT')| (pitch['pitch_type']=='SL')))]

pitch1=pitch1[['ab_id','b_count', 'b_score','on_1b','on_2b','on_3b', 'outs', 'pitch_num', 's_count', 'pitch_type']]
#See if I can get this into 1 step
pitch2=pd.get_dummies(pitch1['pitch_type'])
#pitch1=pd.concat([pitch1, pitch2], axis=1)
pitch1=pitch1.drop('pitch_type', axis=1)

# We can compare the results to the means, which give us the default guess rate of prediction.
pitch2['FF'].mean()
pitch2['FT'].mean()
pitch2['CU'].mean()
pitch2['CH'].mean()
pitch2['SL'].mean()


#Decision Tree Vizualization is commented out because it became unwieldy and after seeing it, it appeared unnecessary. 
"""from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot """

features= list(pitch1.columns)
features

X_train, X_test, y_train, y_test=train_test_split(pitch1, pitch2['FF'], test_size=0.3)
dtree.fit(X_train, y_train)
pred=dtree.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
# We can predict Four Seam Fastballs with 51% accuracy. Four Seam fastballs were thrown 44% of the time.
"""dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())  """
for i in pit:
    
    X_train, X_test, y_train, y_test=train_test_split(pitch1, pitch2[i], test_size=0.3)
    dtree.fit(X_train, y_train)
    pred=dtree.predict(X_test)
    print('For Decision Trees ', i , 'results in : ')
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    #rfc.fit(X_train, y_train)
    #pred2=rfc.predict(X_test)
    #print('For Random Forest Classifier ', i , 'results in : ')
    #print(confusion_matrix(y_test,pred))
    #print(classification_report(y_test,pred))
    """dot_data = StringIO()  
    export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True)

    graph = pydot.graph_from_dot_data(dot_data.getvalue())  
    Image(graph[0].create_png()) """
# Slider can be predicted with 27% accuracy (Four Seam fastballs were thrown 18.4% of the time), Two Seam Fastball can be predicted with 26% accuracy (Two Seam fastballs were thrown 15% of the time),
#Curveball can be predicted with 16% accuracy(Curveballs were thrown 8.6% of the time), and Changeup can be predicted with 21% accuracy (Changeups were thrown 13.6% of the time)
# So we can predict from the game situation better than random, but not that well.

#Classify by Pitcher. Take the 10 pitchers with the most AB.
#This is done to see if individual pitchers are more likely to be predictable than an aggregation of all pitchers.
pitchbat['pitcher_id'].value_counts().head(12)
pitchbat.dropna()
pitchers=[285079, 430935, 434671, 446372, 448306, 451596, 456034, 502042, 502188, 572971]
pitchbat=pitchbat[(((pitchbat['pitch_type']=='FF') | (pitchbat['pitch_type']=='CU')| (pitchbat['pitch_type']=='CH')| (pitchbat['pitch_type']=='FT')| (pitchbat['pitch_type']=='SL')))]
pitchbatdep=pd.get_dummies(pitchbat['pitch_type'])
pitchbatdep=pd.concat([pitchbatdep, pitchbat['pitcher_id']], axis=1)
pitchbat=pitchbat[['pitcher_id','b_count', 'b_score','on_1b','on_2b','on_3b', 'outs', 'pitch_num', 's_count']]


pred_over_rand=np.zeros((10, 5))

for p in pitchers:
    for i in pit:
        pitchloop=pitchbat[(pitchbat['pitcher_id']==p)]
        pitchloopdep=pitchbatdep[(pitchbatdep['pitcher_id']==p)]
        pitchloop=pitchloop.drop('pitcher_id', axis=1)
        pitchloopdep=pitchloopdep.drop('pitcher_id', axis=1)
       
        X_train, X_test, y_train, y_test=train_test_split(pitchloop, pitchloopdep[i], test_size=0.3)
        dtree.fit(X_train, y_train)
        pred=dtree.predict(X_test)
        print('For Pitcher :', p, 'For Decision Trees ', i , 'results in : ')
        print(confusion_matrix(y_test,pred))
        print(classification_report(y_test,pred))
        
        





