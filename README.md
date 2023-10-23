Installation
----

Python 3.9 and 3.10 both work, newer pythons might not be compatable with everything on windows.
To protect your local python environment, use venv, windows example like this:

```bat

python -m venv env

env\Scripts\activate.bat

pip install numpy transformers torch

cd src

python -i chatbot.py
```

After install you can launch like this to use Python's IDLE in the venv:

```bat
env\Scripts\activate.bat

pip install numpy transformers torch

python -m idlelib.idle
```

Either way, you can use the data structure kinda like this:

```python
oracle = ExactVectorIndex(vect_from_str)

#stash a question answer pair
oracle['What is the airspeed velocity of an unladen swallow?'] = 'About 25 miles per hour.'

#stash a fact
fact = 'Birds are tetrachromats. Birds see the world in a 4 dimensional color space.'
oracle[fact] = fact

#get the best answer
print(oracle['tell me about bird vision'])

#get the best N answers
answers = oracle.get_k_nearest('spam', 7)
print(answers[0].value)
print(answers[0].distance)

#if you have a vector from somewhere you can use it directly
x1 = vector_from_string_function('hello')
y1 = oracle[x1]
x2 = answers[1].vector
y2 = oracle[x2]
```

Files
====

chatbot.py
----

Main entry point, it uses the data structures provided in other files for a QA system. Raw data comes from the csv files in /src/data.

embed.py
----

Module for all the text embedding. Uses HuggingFace transformer, see the [Massive Test Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard). We chose [gte-base](https://huggingface.co/thenlper/gte-base), a small model with decent performance.


vectorindex.py
----

This is the vector database. Given a query vector, what document vector is the closest one stored? Allows python dictionary systax for inserts and gets. Also has a method for top-k queries.

questions.log
----

This is not a source file, but it's a record of every question that was asked. It might be useful if you want to write new answers to know what questions people are asking.
