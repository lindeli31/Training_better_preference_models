# Results (val split)

| model                              | method    | stage     |   position_consistency |   accuracy |   position_bias_rate |
|:-----------------------------------|:----------|:----------|-----------------------:|-----------:|---------------------:|
| meta-llama/Llama-3.3-70B-Instruct  | gepa      | baseline  |                  0.707 |      0.611 |                0.293 |
| meta-llama/Llama-3.3-70B-Instruct  | gepa      | optimised |                  0.682 |      0.601 |                0.318 |
| meta-llama/Llama-3.3-70B-Instruct  | opro      | baseline  |                  0.707 |      0.611 |                0.293 |
| meta-llama/Llama-3.3-70B-Instruct  | opro      | optimised |                  0.803 |      0.601 |                0.197 |
| meta-llama/Llama-3.3-70B-Instruct  | opro_tree | baseline  |                  0.707 |      0.611 |                0.293 |
| meta-llama/Llama-3.3-70B-Instruct  | opro_tree | optimised |                  0.727 |      0.606 |                0.273 |
| swiss-ai/Apertus-70B-Instruct-2509 | gepa      | baseline  |                  0.151 |      0.447 |                0.849 |
| swiss-ai/Apertus-70B-Instruct-2509 | gepa      | optimised |                  0.298 |      0.465 |                0.702 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro      | baseline  |                  0.151 |      0.447 |                0.849 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro      | optimised |                  0.651 |      0.543 |                0.348 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro_tree | baseline  |                  0.151 |      0.447 |                0.849 |
| swiss-ai/Apertus-70B-Instruct-2509 | opro_tree | optimised |                  0.586 |      0.505 |                0.414 |