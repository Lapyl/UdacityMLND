import numpy as np
import pandas as pd
import os
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#=================================================================================================================================

def TakeIn(zCsv):
    '''Plot variations of rewards with episodes and steps.'''

    '''Make dataframe from csv file.'''
    zHda = pd.DataFrame(pd.read_csv(zCsv, delimiter=","))

    '''Count steps and sum rewards by episode alpha gamma.'''
    zHdaCol = ['Epis','Alph','Gamm']
    zHdaStp = zHda.groupby(zHdaCol, as_index=False)['Step'].count()
    zHdaIs0 = zHda[zHda.Isou==0].groupby(zHdaCol, as_index=False)['Step'].count()
    zHdaIs1 = zHda[zHda.Isou==1].groupby(zHdaCol, as_index=False)['Step'].count()
    zHdaRwd = zHda.groupby(zHdaCol, as_index=False)['Rewd'].sum()

    '''Combine the above computations into single dataframe.'''
    zHdb = reduce(lambda left,rite: pd.merge(left,rite, on=zHdaCol, how='outer'), [zHdaStp,zHdaIs0,zHdaIs1,zHdaRwd])
    zHdb.columns = ['Episode','Alpha__','Gamma__','StepQty','AddItem','CnvItem','TotRewd']
    zHdb[np.isnan(zHdb)] = 0

    '''Compute average reward for each episode.'''
    zHdb['AvgRewd'] = zHdb['TotRewd']/zHdb['StepQty']
    zHdb['IniIndx'] = zHdb.index.get_values()
    IniRewd = zHdb.iloc[0]['AvgRewd']

    '''Print summary by episode.'''
    print "="*120
    print "Summary by episode alpha gamma (sorted by reward):\n"
    print zHdb

    '''Sort the above combined dataframe by reward.'''
    zHdc = zHdb.sort(columns='AvgRewd', ascending=False)
    zHdc['NewIndx'] = range(0,len(zHdc))
    zHdc = zHdc.set_index('NewIndx')

    '''Find alpha gamma leading to maximum reward.'''
    alpha = zHdc.iloc[0]['Alpha__']
    gamma = zHdc.iloc[0]['Gamma__']
    MaxRewd = zHdc.iloc[0]['AvgRewd']
    MedRewd = zHdc.iloc[int(len(zHdc)/2)]['AvgRewd']
    MinRewd = zHdc.iloc[len(zHdc)-1]['AvgRewd']

    '''Print summary of rewards.'''
    print "="*120
    print "Summary of rewards (average per step):\n"
    ##### "123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789.123456789."
    print " alpha of max reward   gamma of max rewd      Maximum reward   Random act reward      Minimum reward       Median reward"
    print "%20f%20f%20f%20f%20f%20f" % (alpha, gamma, MaxRewd, IniRewd, MinRewd, MedRewd)

    '''Make 3D plot of rewards versus alpha gamma.'''
    zFig = plt.figure()
    ax = zFig.gca(projection='3d')
    ax.plot_trisurf(zHdc['Alpha__'], zHdc['Gamma__'], zHdc['AvgRewd'], cmap=cm.jet) 
    # ax.view_init(elev=20, azim=30)
    # ax.dist = 11
    X, Y = np.mgrid[0:1:20j, 0:1:20j]
    Z = 5
    ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    ax.set_title("Average reward versus alpha gamma")
    ax.set_xlabel("alpha")
    ax.set_ylabel("gamma")
    ax.set_zlabel("Average reward")
    ax.set_xticks((0.0, 0.5, 1.0))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.set_zticks((MinRewd, MedRewd, MaxRewd))
    ay = plt.gca()
    ay.set_axis_bgcolor('DimGray')
    plt.show()
    zFig.savefig(zCsv.replace("H.csv","_RewardByAlphaGamma.jpg"))
    plt.close()

    '''Compute cumulative average rewards against steps in random-action trials.''' 
    zHddInd = zHda[zHda.Epis==0][['Epis','Alph','Gamm','Tryl','Step','Rewd']]
    zHddCum = zHda[zHda.Epis==0][['Epis','Alph','Gamm','Tryl','Rewd']].groupby(['Epis','Alph','Gamm','Tryl']).cumsum()
    zHddCum.columns = ['CumRewd']
    zHdd = pd.concat([zHddInd, zHddCum], axis=1)
    zHdd['AvgRewd'] = np.round(zHdd['CumRewd']/(1+zHdd['Step']), 2)

    '''Print cumulative average rewards against steps in in random-action trials.''' 
    print "="*120
    print "Summary of rewards (average per step):\n"
    print zHdd.head()

    '''Plot cumulative average rewards against steps in in random-action trials.''' 
    zFig = plt.figure(num=1, figsize=(11, 8.5))
    zYxw = np.arange(10) 
    zMat = cm.rainbow(np.linspace(0, 1, len([i + zYxw + (i*zYxw)**2 for i in range(10)])))
    plt.axis([0, 300, -8, 12])
    for xTry in np.arange(10):
        plt.scatter(zHdd[zHdd.Tryl==xTry]['Step'], zHdd[zHdd.Tryl==xTry]['AvgRewd'], label=('Try'+str(xTry)), color=zMat[xTry])
    plt.plot([0,300],[5,5], 'k-', lw=5, label='Bmark')
    plt.title("Avergae cumulative reward versus step in a few tria runs")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc='upper left', mode="expand", ncol=11, borderaxespad=0.)
    plt.show()
    zFig.savefig(zCsv.replace("H.csv","_RewardByStep_Episode00.jpg"))
    plt.close()

    '''Compute cumulative average rewards against steps in Q-action trials.''' 
    zHdeInd = zHda[zHda.Epis==21][['Epis','Alph','Gamm','Tryl','Step','Rewd']]
    zHdeCum = zHda[zHda.Epis==21][['Epis','Alph','Gamm','Tryl','Rewd']].groupby(['Epis','Alph','Gamm','Tryl']).cumsum()
    zHdeCum.columns = ['CumRewd']
    zHde = pd.concat([zHdeInd, zHdeCum], axis=1)
    zHde['AvgRewd'] = np.round(zHde['CumRewd']/(1+zHde['Step']), 2)

    '''Print cumulative average rewards against steps in Q-action trials.''' 
    print "="*120
    print "Summary of rewards (average per step):\n"
    print zHde.head()

    '''Plot cumulative average rewards against steps in Q-action trials.''' 
    zFig = plt.figure(num=1, figsize=(11, 8.5))
    zYxw = np.arange(10) 
    zMat = cm.rainbow(np.linspace(0, 1, len([i + zYxw + (i*zYxw)**2 for i in range(10)])))
    plt.axis([0, 300, -8, 12])
    for xTry in np.arange(10):
        plt.scatter(zHde[zHde.Tryl==xTry]['Step'], zHde[zHde.Tryl==xTry]['AvgRewd'], label=('Try'+str(xTry)), color=zMat[xTry])
    plt.plot([0,300],[5,5], 'k-', lw=5, label='Bmark')
    plt.title("Avergae cumulative reward versus step in a few tria runs")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc='upper left', mode="expand", ncol=11, borderaxespad=0.)
    plt.show()
    zFig.savefig(zCsv.replace("H.csv","_RewardByStep_Episode31.jpg"))
    plt.close()    

#=================================================================================================================================

if __name__ == '__main__':
    '''Run the code.'''

    TakeIn(zCsv = "N:/DevMOOC/UdacityMLnd/P5CapstoneFusion/5Rework4a/fusionH.csv")
