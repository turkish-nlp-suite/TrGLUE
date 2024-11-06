<img src="https://raw.githubusercontent.com/turkish-nlp-suite/.github/main/profile/trgluelogo.png"  width="30%" height="30%">


# TrGLUE - A Comprehensive Turkish NLU Benchmark Dataset

TrQuAD is Turkish SQuAD - Turkish translation of the SQuAD. The dataset can be downloaded from its devoted [HuggingFace repo](https://huggingface.co/datasets/turkish-nlp-suite/TrQuAD).
The dataset is of version 1.1. - there are no empty answer strings. 

More about the dataset preparation can be found in our [research paper](). For the dataset format and more information please visit the HF repo.

### Generating the benchmark

We adopted the original GLUE benchmarking scripts except for `SST-2` task. The original  The benchmarking process can be fired up via `run.sh` . 

Here's the set of hyperparameters we used for benchmarking BERTurk:



```
python3 run_qa.py \
  --model_name_or_path dbmdz/bert-base-turkish-cased \
  --dataset_name turkish-nlp-suite/TrQuAD \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 256 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 128 \
  --doc_stride 32 \
  --output_dir berturk \
  --overwrite_output_dir True
```

The success metrics are same with original SQuAD, exact match and F1 scores. BERTurk achieved scores of `56.09/71.56`.

### Research paper and citation

Coming soon!

