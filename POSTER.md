# Title and Authors
...

# Problem Statement
Order of the information in the prompt has influence on the generated answer.
When we do the preference modeling, how the order of the answer candidates impacts the answer?
How can we mitigate this?

This is important as preference modeling is used for LLM alignment.

This work contains three main contribution:
1. Defining the evaluation criteria of answer consistency
2. Exploring how the order of the answer candidates influences the LLM answer
3. Exploring ways how to mitigate the impact of the ordering on an LLM by (i) changing the LLM instructions and (ii) allowing it to reason.

# Methodology

We ask an LLM to select the best answer from the give two answer candidates (let's call them A and B) given the message history.
The LLM is given three choices:
- "first answer candidate is better"
- "second answer candidate is better"
- "the answer candidates are equally good"

## Measuring consistency and accuracy
On the question level, we track if the LLM was consistent in their preference, and if the chosen answers are correct.

We say that the LLM is consistent on a question if the same answer candidate or the "tie" option was chosen twice under the different ordering.
Then, we report the next metrics:

| Metric                        | Description                                                                                            |
|-------------------------------|--------------------------------------------------------------------------------------------------------|
| `consistency`                 | Fraction of questions where the model was consistent in selection across both answer candidate orders. |
| `bias_toward_first_position`  | Fraction of questions where the model always picked answer candidate that goes first.                  |
| `bias_toward_second_position` | Fraction of questions where the model always picked answer candidate that goes second.                 |
| `tie_inconsistency_rate`      | Fraction of questions where a tie in one condition became a non-tie in the other.                      |

Accuracy metrics:

| Metric          | Description                                                                                                                                |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `hard_accuracy` | Fraction of questions where the model chose correct answers under both permutations.                                                       |                                
| `soft_accuracy` | Fraction of questions where the model chose correct answers in at least one of the permutations.                                           |
| `ab_accuracy`   | Fraction of questions where the model chose correct answer when it was in the first position.                                              |
| `ba_accuracy`   | Fraction of questions where the model chose correct answer when it was in the second position.                                             |
| `accuracy_gap`  | AB accuracy − BA accuracy. A large positive gap means the judge is right in AB partly due to position preference, not quality recognition. |

## Optimisation
We research how the prompt instruction can influencce the answer consistency and accuracy.
We compare (i) prompt instruction optimization algorithms (OPRO, JEPA, and MIPROv2) and (ii) impact of enabling model to reason (TODO: several reasoning strategies?).

# Results
