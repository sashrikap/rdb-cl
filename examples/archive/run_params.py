import yaml

with open("examples/active_template.yaml", "r") as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

locals().update(params)
print(TRUE_W)
