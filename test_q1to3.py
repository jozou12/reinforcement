#!/usr/bin/env python3
"""Small, non-scoring examples: python test_q1to3.py [q1 q2 q3].

These examples use different inputs from the staff grading cases and cover
only a few behaviors. Q2 checks answer format, not policy correctness.
Use Gridworld to inspect values/policies and Crawler to inspect learned motion.
"""
import argparse
import math
from numbers import Real
import random

import analysis
import mdp
from qlearningAgents import QLearningAgent
from valueIterationAgents import ValueIterationAgent


class IncompleteExample(Exception):
    pass


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    check(math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-8),
          '%s: expected %s, received %s' % (label, expected, actual))


class TinyJourney(mdp.MarkovDecisionProcess):
    """A short journey with an immediate exit and a more rewarding detour."""
    def getStates(self):
        return ['junction', 'launch', 'terminal']

    def getStartState(self):
        return 'launch'

    def getPossibleActions(self, state):
        return {'launch': ('advance', 'stop'), 'junction': ('finish',)}.get(state, ())

    def getTransitionStatesAndProbs(self, state, action):
        nextState = 'junction' if (state, action) == ('launch', 'advance') else 'terminal'
        return [(nextState, 1.0)]

    def getReward(self, state, action, nextState):
        return {('launch', 'advance'): -0.25,
                ('launch', 'stop'): 2.0,
                ('junction', 'finish'): 7.5}[(state, action)]

    def isTerminal(self, state):
        return state == 'terminal'


def example_q1():
    world = TinyJourney()
    first = ValueIterationAgent(world, discount=0.6, iterations=1)
    close(first.getValue('launch'), 2.0, 'one-sweep launch value')
    close(first.getValue('junction'), 7.5, 'one-sweep junction value')
    close(first.getQValue('launch', 'advance'), 4.25, 'Q from one-sweep values')
    second = ValueIterationAgent(world, discount=0.6, iterations=2)
    close(second.getValue('launch'), 4.25, 'two-sweep launch value')
    check(second.getPolicy('launch') == 'advance', 'launch should choose advance')
    check(second.getPolicy('terminal') is None, 'terminal policy should be None')


def example_q2():
    for letter in 'abcde':
        name = 'question2' + letter
        result = getattr(analysis, name)()
        if isinstance(result, str) and result == 'NOT POSSIBLE':
            continue  # Valid answer format; this does not verify impossibility.
        check(isinstance(result, tuple) and len(result) == 3,
              name + ' should return (discount, noise, livingReward)')
        if any(value is None for value in result):
            raise IncompleteExample(name + ' still contains None answers')
        check(all(isinstance(value, Real) and math.isfinite(value) for value in result),
              name + ' must contain three finite numbers')
        discount, noise, livingReward = result
        check(0 <= discount <= 1, name + ': discount should lie in [0, 1]')
        check(0 <= noise <= 1, name + ': noise should lie in [0, 1]')


def example_q3():
    actions = {'room': ('inspect', 'leave'), 'next': ('finish',), 'done': ()}
    agent = QLearningAgent(actionFn=actions.__getitem__, alpha=0.4, gamma=0.6)
    close(agent.getQValue('room', 'leave'), 0.0, 'unseen Q-value')
    # First establish a successor value of 2.5, then update the room twice.
    agent.update('next', 'finish', 'done', 6.25)
    agent.update('room', 'inspect', 'next', 7.5)
    close(agent.getQValue('room', 'inspect'), 3.6, 'first room update')
    agent.update('room', 'inspect', 'next', 7.5)
    close(agent.getQValue('room', 'inspect'), 5.76, 'second room update')
    close(agent.computeValueFromQValues('room'), 5.76, 'room value')
    check(agent.computeActionFromQValues('room') == 'inspect',
          'the greedy room action should be inspect')
    agent.setEpsilon(0.0)
    check(agent.getAction('room') == 'inspect', 'provided action selection should use your policy')
    check(agent.getAction('done') is None, 'terminal action should be None')


EXAMPLES = {
    'q1': ('small value-iteration journey', example_q1),
    'q2': ('answer format only; inspect policies in Gridworld', example_q2),
    'q3': ('two Q-learning updates', example_q3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('questions', nargs='*', metavar='qN',
                        help='optional selection from q1, q2, q3')
    args = parser.parse_args()
    if any(question not in EXAMPLES for question in args.questions):
        parser.error('questions must be selected from q1, q2, q3')
    selected = args.questions or list(EXAMPLES)
    print('Small examples only. No points assigned; passing does not establish full correctness.')
    failed = 0
    randomState = random.getstate()
    try:
        for question in selected:
            label, example = EXAMPLES[question]
            random.seed(53705)
            try:
                example()
            except (SystemExit, IncompleteExample) as error:
                detail = 'required method is unfinished (see message above)' if isinstance(error, SystemExit) else str(error)
                print('[INCOMPLETE] %s: %s' % (question, detail))
                failed += 1
            except Exception as error:
                print('[FAIL] %s: %s: %s' % (question, type(error).__name__, error))
                failed += 1
            else:
                print('[PASS] %s: %s' % (question, label))
    finally:
        random.setstate(randomState)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
