"""
Lecture 0
0. Search - finding a solution to a problem

DEFINITIONS:
agent - entity that perceives its environment and acts upon that environment
state - a configuration of the agent and its environment
initial state - state from which the search algorithm starts
actions - choices that can be made in a state. defined as a function Action(s) that returns the set of actions that can be executed in state s
transition model - a description of what state results from performing any applicable action. defined as Result(s,a) returning state resulting
                    performing an action
state space - the set of all states reachable from the initial state by any sequence of actions. can be visualized as a directed graph with states
                    represented as nodes, and actions, represented as arrows between nodes (directed graph)
goal test - the condition that determines whether a given state is a goal state
path cost - numerical cost associated with a given path (weighted graph)
optimal solution - a solution that has the lowest path cost among all solutions
node - a data structure that keeps track of 
            - a state
            - a parent (node that generated this node)
            - an action (action applied to parent to get node)
            - a path cost (from initial state in node)
frontier - represents all the things that could be explored next

APPROACH:
start with a frontier that contains the initial state
repeat:
    - if the frontier is empty, then no solution
    - remove a node from the frontier to consider
    - if node contains goal state, return the solution
    - expand node, add resulting nodes to the frontier (look at neighbors of the node)
but what if the parent state is part of the neighboring states that you can go to?

REVISED APPROACH:
start with a frontier that contains the initial state
start with an empty explored set
repeat:
    - if the frontier is empty, then no solution
    - remove a node from the frontier to consider
    - if node contains goal state, return the solution
*** - add the node to the explored state ***
    - expand node, add resulting nodes to the frontier if they aren't already in the frontier or the explored set
but how do we know which node from the frontier should we remove?

DEPTH-FIRST SEARCH (DFS):
managed as a stack data structure
pros: at best, this algorithm is fastest, ie get lucky and chooses the right path
cons: it is possible that the solution isn't optimal, at worst, this algorithm will explore every possible path before finding the solution
code:
"""
# define the function that removes a node from the frontier and return it
def remove(self):
    #terminate the search if the frontier is empty because there is no solution
    if self.empty():
        raise Exception("empty function")
    else:
        #save the last item in the list
        node = self.frontier[-1]
        #save all the items on the list besides the last node
        self.frontier = self.frontier[:-1]
        return node
"""
BREADTH-FIRST SEARCH (BFS):
managed as a queue data structure
pros: guaranteed to find the optimal solution
cons: algorithm is almost guaranteed to take longer than the minimal time to run, at worst, this algo takes the longest possible time to run
code:
"""
#define the function that removes a node from the frontier and return it
def remove(self):
    #terminate the search if the frontier is empty because there is no solution
    if self.empty():
        raise Exception("empty frontier")
    else:
        # save the oldest item on the list
        node = self.frontier[0]
        #save all the items on the list besides the first one
        self.frontier = self.frontier[1:]
        return node
"""
GREEDY BEST-FIRST SEARCH:
BFS and DFS are both uninformed search algorithms (don't utilize knowledge about the problem)
Greedy best-first search expands the node that is the closest to the goal, as determined by a heuristic function h(n) that estimates how close
the goal the next node is, but it can be mistaken.
The efficiency of this search depends on how good h(n) is

for example: Manhattan distance heuristic function - ignores walls in a maze and counts how many steps up, down, or next to to get to the goal

*** it is possible that an uninformed search will provide a better solution, but it is less likely to do so than an informed algorithm ***

54:45
"""

