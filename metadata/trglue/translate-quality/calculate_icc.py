import pingouin as pg
import pandas as pd
import json

datasets = ["mnli",  "mrpc",  "qnli",  "qqp",  "rte",  "stsb"]

dirs = ["test", "train", "valid"]

annots = ["gpt4", "gemini"]

for dataset in datasets:
  gpt_annots = []
  gemini_annots = []
  for dirname in dirs:
        filen1 = dataset + "/results/" + dirname + "/gpt4.jsonl"
        filen2 = dataset + "/results/" + dirname + "/gemini.jsonl"
        with open(filen1, "r") as infile:
            annotjs = json.load(infile)
            all_labelsgpt = [annota["rating"] for annota in annotjs]
        with open(filen2, "r") as infile:
            annotjs = json.load(infile)
            all_labelsgemini = [annota["rating"] for annota in annotjs]

        len1 = len(all_labelsgpt)
        len2 = len(all_labelsgemini)
        if len1 == len2:
            pass
        else:
          m = min(len1, len2)
          all_labelsgpt = all_labelsgpt[:m]
          all_labelsgemini = all_labelsgemini[:m]

        gpt_annots += all_labelsgpt
        gemini_annots += all_labelsgemini



  data = pd.DataFrame({
    "Annotator_1": gpt_annots,
    "Annotator_2": gemini_annots
  })
  data["instance"] = ["Instance_" + str(i) for i in range(len(gpt_annots))]

  # Melt data into long format for ICC calculation
  data_long = data.melt(id_vars=["instance"], var_name="annotator", value_name="rating")

  # Calculate ICC
  icc = pg.intraclass_corr(data=data_long, targets="instance", raters="annotator", ratings="rating")
  print(icc, dataset)
  print("===================")

