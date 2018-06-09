###OBJECTIVE:

This is a Capstone Project for Udacity Machine Learning Nanodegree.

###PROBLEM:

A controlled thermonuclear fusion reactor should have a feedback mechanism, somewhat similar to the steam engine’s speed-controlling valve. Depending upon the levels and temperatures of items, this feedback mechanism should change the settings of various parameters affecting the reactions.

This project is an attempt towards a possible design for a feedback mechanism, which can improve and control dynamics of reactions in a controlled thermonuclear fusion reactor.

The project makes use of reinforcement learning, which is based on concepts of actions, rewards, and Q values. Apart from some basic concepts of and assumptions for reinforcement learning, there are no datasets involved here.

###Simplification

Instead of dealing with the complicated problem of controlled thermonuclear fusion, let us deal with a simple and generic problem of chemical and nuclear reactions.

Suppose that we have six types of items: items 1, 2, 3, 4, 5, and 6. It is possible to convert item 1 into item 2 (by using changer 12) or item 4 or item 6, item 2 into item 1 or item 3, item 4 into item 1 or item 5, item 5 into item 4 or item 6. Other conversions are not possible. We want to enhance the production of desired item 6 and avoid the production of waste item 3.

###Q-Learning Problem

In the above problem, suppose the environment assigns the agent a reward of $10 when the desired item 6 is produced and a penalty of $10 when the waste item 3 is produced.

Suppose the agent is not informed upfront about the reward scheme. Initially, therefore, the agent has to make random moves. As it goes through a series of steps, it receives rewards and penalties, learns by using reinforcement learning, and starts making calculated moves. 

Input items and the reward system make up the environment, and changers (converters of items) make up the agent. Input items constitute states, and changers constitute actions.

###Two-Entity Problem

Let us assume that two entities of item 1 or 2 or 3 or 4 or 5 or 6 can co-exist at any given time.

If we abbreviate items “i & j” as “ij”, possible input items and possible output items are: 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 33, 34, 35, 36, 44, 45, 46, 55, 56, and 66.

If we abbreviate changers “ij to mn” as “ijmn”, possible changers are: 1122, 1124, 1126, 1142, 1144, 1146, 1166, 1221, 1223, 1241, 1234, 1216, 1236, 1323, 1334, 1336, 1421, 1425, 1441, 1445, 1416, 1456, 1524, 1526, 1544, 1546, 1566, 1626, 1646, 1666, 2211, 2213, 2233, 2313, 2333, 2411, 2415, 2413, 2435, 2514, 2516, 2534, 2536, 2616, 2636, 3431, 3435, 3534, 3536, 4411, 4415, 4455, 4514, 4516, 4545, 4556, 4616, 4656, 5544, 5546, 5566, 5646, 5666.

###Simulation

Capstone-LalitAPatel-R4-Part1Simulate.py is an illustrative python code to simulate this case of two entities.

Part1_Screen.jpg is a screenshot taken manually from the screen from this simulation. Part1_Text.txt is the text of the same screen saved as a txt file.

The simulation code generates fusionQ.csv and fusionH.csv files. fusionQ.csv is the Q table of updated q values, and fusionH.csv is the history table of simulation records.

###Analysis of Simulation Results

Capstone-LalitAPatel-R4-Part2Analyze.py is an illustrative python code to analyze fusionH.csv file of history table of the simulation records.

Part2_Screen.jpg is a screenshot taken manually from the screen from a typical simulation session. Part2_Text.txt is the text of the same screen saved as a txt file.

This analysis code generates fusion_RewardByAlphaGamma.jpg, fusion_RewardByStep_Episode00.jpg, and fusion_RewardByStep_Episode31.jpg files.