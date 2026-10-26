import subprocess

print("Starting ML Pipeline.. \n")

subprocess.run(["python", "pipeline/preprocess.py"])
subprocess.run(["python", "pipeline/train.py"])
subprocess.run(["python", "pipeline/predict.py"])


print("Executed Successfully \n")
