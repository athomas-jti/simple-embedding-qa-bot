from time import time
start = time()
print('loading ...', end='')
from embed import vect_from_str
from vectorindex import ExactVectorIndex
import csv

def say_chatbot():
    print('Hello. I am a question answering bot.')
    print("I don't have conversational memory, so I only see one question at a time.")
    print("If you don't like an answer I can always give another, just say:")
    print('-->next')
    print('and I will give you the next match.')
    print("If you don't want any more answers, you can quit like:")
    print('--->quit')
    print('You can get this message again by asking:')
    print('-->?')
    print()

def say_item(it):
    print(it.value)
    score = 100 * (1-it.distance)
    print(f'[score:{score: >3.0f}]')
    print()

oracle = ExactVectorIndex(vect_from_str)

with open('data/qa.csv', newline='') as csv_file:
    for q, a in csv.reader(csv_file):
        oracle[q] = f'Q:\n{q}\nA:\n{a}'
with open('data/facts.csv', newline='') as csv_file:
    for fact in csv.reader(csv_file):
        fact = fact[0]
        oracle[fact] = fact

elapsed = time() - start
print(f'... done! (startup in {elapsed:.1f} seconds)')
print()

say_chatbot()
user_questions = []
items = []
try:
    while True:
        s = input('--> ')
        s = s.strip()
        if s == 'next':
            if items:
                say_item(items.pop(0))
            else:
                say_chatbot()
        elif s == '':
            if items:
                say_item(items.pop(0))
            else:
                print('nothing else')
        elif s == '?':
            say_chatbot()
        elif s == 'quit' or s == 'exit':
            break
        else:
            user_questions.append(s)
            items = oracle.get_k_nearest(s, 15)
            say_item(items.pop(0))
except:
    pass

with open('questions.log', 'at') as log:
    log.write('\n'.join(user_questions))
    log.write('\n')
    
