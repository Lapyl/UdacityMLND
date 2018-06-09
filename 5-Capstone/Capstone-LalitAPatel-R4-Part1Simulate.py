'''
Udacity Machine Learning Nanodegree Capstone Project
Reinforcement Learning to Improve Reaction Dynamics
Lalit A Patel
'''

#=================================================================================================================================
import numpy as np
import pandas as pd
import time
import os
import random
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#=================================================================================================================================
def SessionStart():
    '''Start a session.'''

    '''Get start time.'''
    zTst = time.time()

    '''Create a blank H dataframe.'''
    zHin = 0
    zHco = ['Epis','Alph','Gamm','Hind','Tryl','Step','Isou','Item','Csou','Chgr','Outp','Rewd','Qind','Qini','Qnew','Time']
    zHda = pd.DataFrame(columns=zHco)
    
    '''Create a blank Q dataframe.'''
    zQco = ['Qind','Item','Chgr','Outp','Rewd','Qini','Qnew']
    zQda = pd.DataFrame(columns=zQco)

    '''Populate the Q dataframe.'''
    zQda = zQda.append(pd.DataFrame([[  0, 11, 1124, 24,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  1, 11, 1126, 26,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  2, 11, 1142, 42,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  3, 11, 1144, 44,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  4, 11, 1146, 46,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  5, 11, 1166, 66,  20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  6, 12, 1216, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  7, 12, 1221, 21,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  8, 12, 1223, 23, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[  9, 12, 1234, 34, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 10, 12, 1236, 36,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 11, 12, 1241, 41,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 12, 13, 1323, 23, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 13, 13, 1334, 34, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 14, 13, 1336, 36,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 15, 14, 1416, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 16, 14, 1421, 21,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 17, 14, 1425, 25,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 18, 14, 1441, 41,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 19, 14, 1445, 45,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 20, 14, 1456, 56,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 21, 15, 1524, 24,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 22, 15, 1526, 26,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 23, 15, 1544, 44,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 24, 15, 1546, 46,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 25, 15, 1566, 66,  20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 26, 16, 1626, 26,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 27, 16, 1646, 46,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 28, 16, 1666, 66,  20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 29, 22, 2211, 11,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 30, 22, 2213, 13, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 31, 22, 2233, 33, -20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 32, 23, 2313, 13, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 33, 23, 2333, 33, -20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 34, 24, 2411, 11,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 35, 24, 2413, 13, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 36, 24, 2415, 15,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 37, 24, 2435, 35, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 38, 25, 2514, 14,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 39, 25, 2516, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 40, 25, 2534, 34, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 41, 25, 2536, 36,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 42, 26, 2616, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 43, 26, 2636, 36,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 44, 34, 3431, 31, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 45, 34, 3435, 35, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 46, 35, 3534, 34, -10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 47, 35, 3536, 36,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 48, 44, 4411, 11,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 49, 44, 4415, 15,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 50, 44, 4455, 55,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 51, 45, 4514, 14,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 52, 45, 4516, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 53, 45, 4545, 45,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 54, 45, 4556, 56,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 55, 46, 4616, 16,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 56, 46, 4656, 56,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 57, 55, 5544, 44,   0, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 58, 55, 5546, 46,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 59, 55, 5566, 66,  20, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 60, 56, 5646, 46,  10, 0, 0 ]], columns=zQco), ignore_index=True)
    zQda = zQda.append(pd.DataFrame([[ 61, 56, 5666, 66,  20, 0, 0 ]], columns=zQco), ignore_index=True)

    return [zHin, zHco, zHda, zQda, zTst]

#=================================================================================================================================
def EpisodeStart():
    '''Start a sub-session.'''

    print "="*120
    print "      Epis  Alph  Gamm      Hind  Tryl  Step      Isou  Item  Csou      Chgr  Outp  Rewd      Qind  Qini  Qnew      Time"
    print "="*120

#=================================================================================================================================
def TrialStart(zQda):
    '''Start a trial.'''

    zQda['Qini'] = 0
    zQda['Qnew'] = 0

    zStp = 0
    zIso = 0
    zItm = 0
    zQio = 99
    
    return [zStp, zIso, zItm, zQio, zQda]

#=================================================================================================================================
def StepAdd(zEpi, alpha, gamma, zHin, zHco, zHda, zTry, zStp, zIso, zItm, zQio, zQda, zTst):
    '''Add a step record in H array.'''

    '''Find Q array records corresponding to the state.'''    
    zQdc = zQda[zQda.Item==zItm]

    '''Find an appropraite action after checking Q array.'''
    if len(zQdc)>0:

        '''Add a blank step record.'''
        zHda = zHda.append(pd.DataFrame([[zEpi,alpha,gamma,zHin,zTry,zStp,zIso,zItm,0,0,0,0,0,0,0,time.time()-zTst]], columns=zHco), ignore_index=True)

        if zEpi == 0:

            zQdc = zQdc.sample(1)

            zHda.at[zHin,'Chgr'] = zQdc['Chgr'].iloc[0]
            zHda.at[zHin,'Outp'] = zQdc['Outp'].iloc[0]
            zHda.at[zHin,'Rewd'] = zQdc['Rewd'].iloc[0]

        else:

            zQva = zQdc['Qnew'].max()
            zQdc = zQdc[(zQdc.Qnew==zQva)]
            zCso = len(zQdc)
            if zCso > 1: zQdc = zQdc.sample(1)

            zHda.at[zHin,'Csou'] = zCso
            zHda.at[zHin,'Chgr'] = zQdc['Chgr'].iloc[0]
            zHda.at[zHin,'Outp'] = zQdc['Outp'].iloc[0]
            zHda.at[zHin,'Rewd'] = zQdc['Rewd'].iloc[0]
            zHda.at[zHin,'Qind'] = zQdc['Qind'].iloc[0]
            zHda.at[zHin,'Qini'] = zQva
            zHda.at[zHin,'Qnew'] = zQva

            if zQio in np.arange(31):
                zQda['Qini'][zQio] = zQda['Qnew'][zQio]
                zQda['Qnew'][zQio] = int((1-alpha) * zQda['Qnew'][zQio] + alpha * (zHda.at[zHin-1,'Rewd'] + gamma * zQva))
                zHda.at[zHin-1,'Qnew'] = zQda['Qnew'][zQio]

        '''Print H array record and create a plot corresponding to the step.'''
        print "%10i%6i%6i%10i%6i%6i%10i%6i%6i%10i%6i%6i%10i%6i%6i%10i" % (
            zHda.at[zHin,"Epis"],100*zHda.at[zHin,"Alph"],100*zHda.at[zHin,"Gamm"],zHda.at[zHin,"Hind"],zHda.at[zHin,"Tryl"],zHda.at[zHin,"Step"],
            zHda.at[zHin,"Isou"],zHda.at[zHin,"Item"],zHda.at[zHin,"Csou"],zHda.at[zHin,"Chgr"],zHda.at[zHin,"Outp"],
            zHda.at[zHin,"Rewd"],zHda.at[zHin,"Qind"],zHda.at[zHin,"Qini"],zHda.at[zHin,"Qnew"],zHda.at[zHin,"Time"])

        zQio = zQdc['Qind'].max()
        zItm = int(zHda.at[zHin, 'Outp'])

    return [zHda, zItm, zQio, zQda]

#=================================================================================================================================
def AGSelect(zHda):
    '''Select optimal alpha and gamma.'''

    zRwd = zHda.groupby(['Epis','Alph','Gamm'], as_index=False)['Rewd'].sum()
    print "="*120
    print "Total rewards by episode (alpha gamma)"
    print zRwd

    zRwe = zRwd.sort(columns='Rewd', ascending=False)
    zRwe['Nind'] = range(0,len(zRwe))
    zRwe = zRwe.set_index('Nind')
    print "="*120
    print "Total rewards by episode (alpha gamma) sorted by total rewards"
    print zRwe

    alpha = zRwe.iloc[0]['Alph']
    gamma = zRwe.iloc[0]['Gamm']
    print "="*120
    print "alpha gamma leading to maxmim total rewards"
    print alpha
    print gamma

    return [alpha, gamma]

#=================================================================================================================================
def main(yEpr, yTmr, yTmo):
    '''Define the main function.'''

    '''Run a session.'''
    [yHin, yHco, yHda, yQda, yTst] = SessionStart()
    yHqi = 0
    
    '''Run episodes of different alpha & gamma values.'''
    for yEpi in np.arange(yEpr+2):

        if yEpi==0:
            [alpha, gamma] = [ 0.0, 0.0 ]
            yTmx = yTmr
        elif yEpi<yEpr+1:        
            [alpha, gamma] = [ random.randrange(0,100,1)/100.00, random.randrange(0,100,1)/100.00 ]
            yTmx = yTmr
        else:
            [alpha, gamma] = AGSelect(zHda=yHda)
            yTmx = yTmo

        EpisodeStart()
        
        '''Run trials for averaging.'''
        for yTry in np.arange(yTmx):

            [yStp, yIso, yItm, yQio, yQda] = TrialStart(zQda=yQda)
            
            '''Run steps within a trial.'''
            while yStp < 300:

                if yItm in [11,12,13,14,15,16,22,23,24,25,26,34,35,44,45,46,55,56]:
                    yIso = 1
                else:
                    yIso = 0
                    ySam = int(10 * random.random())
                    yItm = int([11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 33, 34, 35, 36, 44, 45, 46, 55, 56, 66][ySam])
                    yQio = 99

                '''Add a step record.'''
                [yHda, yItm, yQio, yQda] = StepAdd(zEpi=yEpi, alpha=alpha, gamma=gamma, zHin=yHin, zHco=yHco, zHda=yHda, zTry=yTry, zStp=yStp, zIso=yIso, zItm=yItm, zQio=yQio, zQda=yQda, zTst=yTst)

                '''Increase the step value, subject to a limit.'''            
                yHin += 1
                yStp += 1

    '''Save csv files of records of H and Q arrays.'''    
    if os.path.exists("fusionH.csv"): os.remove("fusionH.csv")
    yHda.to_csv("fusionH.csv", sep=",", encoding='utf-8', index=True, mode='a', header=True)

    if os.path.exists("fusionQ.csv"): os.remove("fusionQ.csv")
    yQda.to_csv("fusionQ.csv", sep=",", encoding='utf-8', index=True, mode='a', header=True)

    print "Done"

#=================================================================================================================================
if __name__ == '__main__':
    '''Run the code.
       yEpr: Number of random alpha-gammas to be tried
       yTmr: Number of trials to be run for random alpha-gamma
       yTmo: Number of trials to be run for optimal alpha-gamma'''

    main(yEpr=30, yTmr=30, yTmo=60)
