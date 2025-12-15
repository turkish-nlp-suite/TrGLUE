<img src="https://raw.githubusercontent.com/turkish-nlp-suite/.github/main/profile/trgluelogo.png"  width="30%" height="30%">


# TrGLUE - A Natural Language Understanding Benchmark for Turkish
TrGLUE is a NLU benchmarking dataset for Turkish. As the name suggests, it's GLUE benchmarking dataset for Turkish language. You can download the datasets from the
[HuggingFace repo](https://huggingface.co/datasets/turkish-nlp-suite/TrGLUE). For more information about the dataset, the tasks, data curation and more please visit the HF repo.


### Benchmarking
Benchmarking code can be find under `scripts/`. To run a single task run `run_single.sh`:

```
#!/bin/bash

# Pick the task accordingly, here we pick COLA for an example

python3 run_trglue.py \
--model_name_or_path dbmdz/bert-base-turkish-cased \ 
--task_name cola \ 
--max_seq_length 128 \ 
--output_dir berturk \ 
--num_train_epochs 5 \ 
--learning_rate 2e-5 \ 
--per_device_train_batch_size 128 \ 
--per_device_eval_batch_size  128 \ 
--do_train \ 
--do_eval \ 
--do_predict

```

Available task names are:


- cola
- mnli
- sst2
- mrpc
- qnli
- qqp
- rte
- stsb

To run all the tasks in order, please run `run_all.sh`. Benchmarking for BERTurk model and a handful LLMs can be found under the HF repo and the research paper. Here are the batch size and learning rates to replicate the paper results:

```
RTE, STS-B, MRPC: batch size 16, lr 3e-5
All other datasets: batch size 128, lr 2e-5
```



### Research paper and citation
Coming soon!
