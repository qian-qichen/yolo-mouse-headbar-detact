import numpy as np
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

def to_jsonable(obj):

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_jsonable(value) for key, value in obj.items()}
    if hasattr(obj, '__dict__'):
        return to_jsonable(vars(obj))
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")