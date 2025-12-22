import yaml
from dataclasses import is_dataclass, fields

def load_yaml_as_dataclass(yaml_path, dataclass_type):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return dataclass_from_dict(dataclass_type, data)

def dataclass_from_dict(cls, data: dict):
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        name = f.name
        if name not in data:
            continue
        val = data[name]
        if val is None:
            if f.default is None:
                kwargs[name] = None
            else:
                raise ValueError(f"Field {name} is None but required a value.")
            continue
        field_type = f.type
        if is_dataclass(field_type):
            kwargs[name] = dataclass_from_dict(field_type, val)
        else:
            kwargs[name] = val
    return cls(**kwargs)