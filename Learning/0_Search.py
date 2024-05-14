"""
Lecture 0
0. Search: finding a solution to a problem

def
agent - entity that perceives its environment and acts upon that environment
state - a configuration of the agent and its environment
initial state - state from which the search algorithm starts
actions - choices that can be made in a state. defined as a function Action(s) that returns the set of actions that can be executed in state s
transition model - a description of what state results from performing any applicable action. defined as Result(s,a) returning state resulting
                    performing an action
state space - the set of all states reachable from the initial state by any sequence of actions. can be visualized as a directed graph with states
                    represented as nodes, and actions, represented as arrows between nodes
goal test - the condition that determines whether a given state is a goal state
path cost - numerical cost associated with a given path
"""