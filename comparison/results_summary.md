# Results (val split)

| model                              | method    | stage     |   position_consistency |   accuracy |   position_bias_rate |
|:-----------------------------------|:----------|:----------|-----------------------:|-----------:|---------------------:|
| meta-llama/Llama-3.3-70B-Instruct  | gepa      | baseline  |                  0.712 |      0.606 |                0.288 |
| meta-llama/Llama-3.3-70B-Instruct  | gepa      | optimised |                  0.682 |      0.596 |                0.318 |
| meta-llama/Llama-3.3-70B-Instruct  | opro      | baseline  |                  0.712 |      0.606 |                0.288 |
| meta-llama/Llama-3.3-70B-Instruct  | opro      | optimised |                  0.692 |      0.558 |                0.308 |
| meta-llama/Llama-3.3-70B-Instruct  | opro_tree | baseline  |                  0.712 |      0.606 |                0.288 |
| meta-llama/Llama-3.3-70B-Instruct  | opro_tree | optimised |                  0.854 |      0.614 |                0.146 |
| swiss-ai/Apertus-70B-Instruct-2509 | gepa      | baseline  |                  0.157 |      0.447 |                0.843 |
| swiss-ai/Apertus-70B-Instruct-2509 | gepa      | optimised |                  0.333 |      0.460 |                0.667 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro      | baseline  |                  0.157 |      0.447 |                0.843 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro      | optimised |                  0.687 |      0.528 |                0.313 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro_tree | baseline  |                  0.157 |      0.447 |                0.843 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro_tree | optimised |                  0.589 |      0.463 |                0.411 |