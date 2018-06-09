import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import os # Package for files etc
import numpy as np # Package for maths
import pandas as pd # Package for dataframes
import matplotlib.pyplot as plt # Package for plots
    
# Create a blank dataframe for cab's driving data.
if os.path.exists("cabdata.csv"): os.remove("cabdata.csv") # Remove cabdata.csv file if it exists.
cabdata = pd.DataFrame(columns=['Stage','Session','Trial','Step','Alpha','Gamma','Deadline','NextWay','Left','Right','Oncoming','Light','Action','Reward','TotRewd','TotPass','TotFail']) # Create a blank dataframe.
cabdata.to_csv("cabdata.csv", sep=",", encoding='utf-8', index=False) # Save it as a csv file.

# Define LearningAgent in terms of session number session, learning rate alpha, and discount factor gamma.
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, stage, session, alpha, gamma):

        # Initiate variables.
        super(LearningAgent, self).__init__(env)  # Set self.env=env, state=None, next_waypoint=None, and a default color.
        self.color = 'red'  # Override color.
        self.planner = RoutePlanner(self.env, self)  # Use route planner to get next_waypoint.

        # TODO: Initialize any additional variables.

        # Globalize stage, session, alpha, and gamma variables by treating them as attributes of self.
        self.stage = stage # Stage number to differentiate method of selecting agent's actions
        self.session = session # Session number to differentiate alpha and gamma
        self.alpha = alpha # Learning rate (alpha)
        self.gamma = gamma # Discount factor (gamma)

        # Introduce trial, step (time), trial, and other parameters.
        self.trial = 0 # Trial number to differentiate destination
        self.step = 0 # Step or time number to differentiate agent's location
        self.totRewd = 0 # Cumulative reward
        self.totPass = 0 # Cumulative pass count
        self.totFail = 0 # Cumulative fail count

        # Get a list of valid actions. Create an empty Q table.
        self.validActions = Environment.valid_actions # Environment.valid_actions lists actions allowed for various environments. Use self. for convenience.

        # For stage 2 and 3, set up Q table for all possible states and actions.
        self.Q = {} # Start with an empty Q table.
        if self.stage > 1: # Stage 1 does not use Q table.
            for left in self.validActions: # Loop for left traffic.
                for right in self.validActions: # Loop for right traffic.
                    for oncoming in self.validActions: # Loop for oncoming traffic.
                        for light in ['green', 'red']: # Loop for traffic light.
                            for next_waypoint in self.validActions[1:]: # Loop for next (intended) waypoint.
                                self.state = (left, right, oncoming, light, next_waypoint) # Define state as a tuple of left, right, oncoming, light, and next-waypoint.
                                self.Q[self.state] = {} # Create indexes.
                                for action in self.validActions: # Loop for action.
                                     self.Q[self.state][action] = 0 # Create an array of q values for states and actions. Set q values to 0. Any other value is ok.

    def reset(self, destination=None):

        # Prepare for a new trip and define new destination.
        self.planner.route_to(destination) # Route planner sets a destination for this trial (trip).

        # TODO: Reset any additional variables.
        
        # Reset trial and step numbers.
        self.trial += 1 # Each session consists of several trials. Increase trial number by 1 at each reset.
        self.step = 0 # Each trial will has multiple steps, which are numbered sequentially, strating from 0.

        # Introduce previous state, action, and reward for updating Q table later on.
        self.prevState = None # Previous state
        self.prevAction = None # Previous action
        self.prevReward = None # Previous reward

        # Set up header to print step-by-step results in a trial.
        print "    Stage  Session    Trial     Step    Alpha    Gamma Deadline  NextWay     Left    Right Oncoming    Light   Action   Reward  TotRewd  TotPass  TotFail"
        print "|========|========|========|========|========|========|========|========|========|========|========|========|========|========|========|========|========"

    def update(self, step):

        global cabdata # Make cabdata dataframe global.

        self.step += 1 # Increase step (time) number by 1.
        
        # TODO: Gather self.inputs and update state.
        self.inputs = self.env.sense(self) # Traffic and traffic light
        self.nextWaypt = self.planner.next_waypoint()  # Motion suggested by route planner and also displayed by simulator
        self.deadline = self.env.get_deadline(self) # Time left
        # Define the state in terms of traffic (left, right, oncoming), traffic light, and cab's next waypoint.
        # Deadline is not used here, becuase traffic rules do not care about time for the cab to reach destination. 
        self.state = (self.inputs['left'], self.inputs['right'], self.inputs['oncoming'], self.inputs['light'], self.nextWaypt) # Define state.

        # Define rules for the cab's action for a given state.
        # For stage 1, select a random action.
        if self.stage == 1:
            action = random.choice(self.validActions) # Action as a random action
        # For stage 2 or 3, select action based on maximum q value for possible actions for current state.
        else:
            q = 0 # Start with default q value.
            action = None
            thatAction = None
            for thisAction in self.Q[self.state]: # Loop through all possible actions for this state.
                if (self.Q[self.state][thisAction] > q) or (self.Q[self.state][thisAction] == q and thisAction != None):
                    action = thisAction # Choose this loop's action if it has higher q than previous loop's action.
                    q = self.Q[self.state][thisAction] # Carry forward this loop's q for comparing in next loop.
                    thatAction = action # Carry forward this loop's action for next loop.
                elif self.Q[self.state][thisAction] == q and thisAction == None and thatAction != None:
                    action = thatAction # Choose previous loop's action if this loop's action is None and both have same q.
                    q = self.Q[self.state][thisAction]
                elif self.Q[self.state][thisAction] == q:
                    action = random.choice([thatAction, thisAction]) # Randomly select previous or this loop's action if both hae same q.
                    q = self.Q[self.state][thisAction]
                    thatAction = action

        # Get reward for the action, from the Environment.
        reward = self.env.act(self, action) # Reward

        # Use this reward to accumulate totals.
        self.totRewd += reward # Add current reward to cumulated reward.
        if reward > 5: self.totPass += 1 # If reward exceeds 5, treat this as a Pass and add to cumulated pass count. 
        if reward < 0: self.totFail += 1 # If reward is negative, treat this as a Fail and add to cumulated fail count. 
 
        #  TODO: Learn policy based on self.state, action, reward.
        
        # For state 2 or 3, adjust the Q table. (Stage 1 does not use Q table.)
        if self.stage > 1:
            if self.prevState != None: # For not None previous state
                # UpdatedPreviousQValue = (1-LearningRate)*ExistingPreviousQValue + LearningRate*(PreviousReward + DiscountFactor*CurrentQValue)
                self.Q[self.prevState][self.prevAction] = (1-self.alpha) * self.Q[self.prevState][self.prevAction] + self.alpha * (self.prevReward + self.gamma * self.Q[self.state][action])

        # Carry forward current state/action/reward as previous state/action/reward for next step.
        self.prevState = self.state # Carry forward state.
        self.prevAction = action # Carry forward action.
        self.prevReward = reward # Carry forward reward.
        
        # Print current step's result.
        print "%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s" % (self.stage, self.session, self.trial, self.step, self.alpha, self.gamma, self.deadline, self.nextWaypt,
            self.inputs['left'], self.inputs['right'], self.inputs['oncoming'], self.inputs['light'], action, reward, self.totRewd, self.totPass, self.totFail)       

        # Update cabdata dataframe and cabdata csv file.
        # Define dataframe of current step's results.
        newdata = pd.DataFrame([[self.stage, self.session, self.trial, self.step, self.alpha, self.gamma, self.deadline, self.nextWaypt,
            self.inputs['left'], self.inputs['right'], self.inputs['oncoming'], self.inputs['light'], action, reward, self.totRewd, self.totPass, self.totFail]],
            columns=['Stage','Session','Trial','Step','Alpha','Gamma','Deadline','NextWay','Left','Right','Oncoming','Light','Action','Reward','TotRewd','TotPass','TotFail'])
        cabdata = cabdata.append(newdata, ignore_index=True) # Update dataframe by appending dataframe of current step's results.
        newdata.to_csv("cabdata.csv", sep=",", encoding='utf-8', index=False, mode='a', header=False) # Update csv file by adding current step's results.

# Define eval function for evaluating the generated csv file.
def eval():
    """Evaluate the data generated by the agent."""

    print "Evaluation of the data generated by the agent" + " " + "="*50
    zCsv = pd.DataFrame(pd.read_csv("cabdata.csv", delimiter=",")) # Make dataframe from the generated csv file.
    zCsv['TotResu'] = zCsv['TotPass'] - 2*zCsv['TotFail'] # Add TotResu field computed as TotPass-2*TotFail.
    fig = plt.figure() # Introduce a figure for plotting.
    zSeq = 0 # Index for subplots

    for para in ['TotPass','TotFail','TotResu']: # Create loop for TotPass, TotFail, and TotResu parameters.

        zPrA = zCsv[['Alpha','Gamma',para]].groupby(['Alpha','Gamma']).max() # Create dataframe A of maximum values of parameter for alphas & gammas. 
        zPrA = zPrA.reset_index(level=['Alpha','Gamma']) # Reset index in dataframe A.
        print "-"*20 + " Variation of " + para + " with alpha and gamma"
        print zPrA # Print dataframe A.
        zPrB = pd.pivot_table(zPrA, index=['Alpha'], columns=['Gamma'], aggfunc={para:'max'}) # Create dataframe B by crosstabulating dataframe A.
        # zPrB = zPrB.update(zPrB.fillna(value=0)) # Let us keep N/A as is.
        print "-"*20 + " Crosstab of " + para + " with alpha on rows and gamma on columns"
        print zPrB # Print dataframe B.
        zPrC = zPrA.loc[zPrA[para].idxmax()] # Get result C of alpha & gamma for the best parameter.
        print "-"*20 + " Values of alhpa & gamma for best " + para
        print zPrC # Print result C.

        # Make a heatmap of parameter against alpha & gamma.
        zSeq += 1 # Increase subplot index by 1.
        ax = plt.subplot(6,1,zSeq) # Add subplot.
        plt.colorbar(ax.pcolormesh(zPrB)) # Add a colorbar on the side.
        plt.xticks(np.arange(0.5, len(zPrB.columns), 1), zPrB.columns) # Define x axis.
        plt.yticks(np.arange(0.5, len(zPrB.index), 1), zPrB.index) # Define y axis.
        plt.text(0.5, 0.5, 'Max ' + para, ha='center', va='center', fontsize=16, color="r", alpha=.5) # Add legend.
        
        zPrD = zCsv[['Session','Alpha','Gamma','Trial',para]].groupby(['Session','Alpha','Gamma','Trial']).max() # Create dataframe D of values of parameter for sessions, alphas, gammas, trials. 
        zPrD = zPrD.reset_index(level=['Session','Alpha','Gamma','Trial']) # Reset index in dataframe D.
        zPrE = pd.pivot_table(zPrD, index=['Trial'], columns=['Session'], aggfunc={para:'max'}) # Create dataframe E by crosstabulating dataframe D.
        zPrE.reset_index(drop=True) # Drop index in dataframe E.
        print "-"*20 + " Crosstab of " + para + " with trial on rows and session on columns"
        print zPrE # Print dataframe E.
        
        # Make a chart of variation of parameter with trial.
        zSeq += 1 # Increase subplot index by 1.
        ax = plt.subplot(6,1,zSeq) # Add subplot.
        for zCol in np.arange(9): # Loop through 9 columns of dataframe F.
            plt.plot(zPrE[[zCol]]) # Define subplot.

    # Process the figure of subplots.
    fig.savefig('cabdata.png') # Save it as a pmg file.
    plt.show() # Display it on the screen.
    plt.close(fig) # Close when the display is closed.

    # Return alpha and gamma.
    print "Best pass minus fail total is achieved for alpha = " + str(zPrC['Alpha']) + " and gamma = " + str(zPrC['Gamma'])
    return zPrC # Carry forward the verdict.

def run(stage, session, alpha, gamma):
    """Run the agent for a finite number of trials."""
    
    e = Environment()  # Create environment and add some dummy traffic.
    a = e.create_agent(LearningAgent, stage=stage, session=session, alpha=alpha, gamma=gamma)  # Create agent.
    e.set_primary_agent(a, enforce_deadline = (False if stage==0 else True)) # # Set primary agent. Press Esc or close pygame window to quit.
    sim = Simulator(e, update_delay = 0.000001)  # Create simulator.
    sim.run(n_trials = (10 if stage==0 else (300 if stage==2 else 100))) # Run 10/300/100 trials in stage 1/2/3.

if __name__ == '__main__':
    """Run all 3 stages together. Or, run each stages seprately, by commenting run code for other stages."""

    # In stage 1, run the agent by taking randomly selected actions.
    run(stage=1, session=0, alpha=0.00, gamma=0.00)

    # In stage 2, run the agent by taking actions by updating Q table with randomly selected learning rate (alpha) and discount factor (gamma).
    for session in np.arange(201,210,1):
        run(stage=2, session=session, alpha=random.randrange(95,100,1)/100.00, gamma=random.randrange(30,40,1)/100.00)

    # Evaluate the data. Get best alpha and gamma.
    eval()

    # In stage 3, run the agent by taking actions by updating Q table with best alpha and gamma.
    run(stage=3, session=301, alpha=zPrC['Alpha'], gamma=zPrC['Gamma'])
