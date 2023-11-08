import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--name", default=None, type=str, dest="name")
parser.add_argument("--template", default="Transformer_Run15", type=str, dest="template")
parser.add_argument("--dataset", default="", type=str, dest="dataset")
args_parser = parser.parse_args()

name = args_parser.name
template_name = args_parser.template
dataset = args_parser.dataset

with open(f"./runs/{template_name}/config.json") as config_template:
    config_template = config_template.read()
    config_template = config_template.replace(template_name, name)

with open(f"./runs/{template_name}/run.tbi") as tbi_template:
    tbi_template = tbi_template.read()
    tbi_template = tbi_template.replace(template_name, name)

with open(f"./runs/{template_name}/run.sh") as sh_template:
    sh_template = sh_template.read()
    sh_template = sh_template.replace(template_name, name)

os.makedirs(f"./runs/{name}", exist_ok=True)
os.makedirs(f"./runs/{name}/tensorboard", exist_ok=True)

with open(f"./runs/{name}/config.json", "w") as config:
    config.write(config_template)

with open(f"./runs/{name}/run.tbi", "w") as tbi:
    tbi.write(tbi_template)

with open(f"./runs/{name}/run.sh", "w") as sh:
    sh.write(sh_template)
