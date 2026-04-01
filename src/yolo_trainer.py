import os
import argparse
from matplotlib import scale
import yaml
from ultralytics.models.yolo import YOLO
import torch

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

SCALES2INT ={
    'n': 0,
    's': 1,
    'm': 2,
    'l': 3,
    'x': 4

}

def prepare_env(cfg):
    cuda_dev = cfg.get('cuda_visible_devices')
    if cuda_dev:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_dev)
    # ensure project/name exist
    project = cfg.get('project', 'runs')
    name = cfg.get('name', 'exp')
    os.makedirs(os.path.join(project, name, 'weights'), exist_ok=True)
    return project, name

# def load_model(model_cfg):
#     model_yaml = model_cfg.get('yaml')
#     model = YOLO(model_yaml) if model_yaml else None
#     pretrained = model_cfg.get('pretrained')
#     if pretrained and model is not None:
#         model = model.load(pretrained)
#     return model

def run_stage(model_cfg, stage_cfg, top_cfg):
    model_yaml = model_cfg.get('yaml')
    stage_name = stage_cfg.get('name', '<stage>')
    print(f"Starting stage: {stage_name}")
    torch.cuda.empty_cache()
    # scale = model_cfg.get('scale', 'm')
    # if stage has its own pretrained, load it; otherwise try to load last weights if not first stage
    candidate = os.path.join(top_cfg.get('project','runs'), top_cfg.get('name','exp'), 'weights', 'last.pt')
    if 'pretrained' in stage_cfg and stage_cfg['pretrained']:
        print(f"Loading pretrained for stage {stage_name}")
        pretrained = stage_cfg['pretrained']
    elif os.path.isfile(candidate):
        print(f"Loading last weights for stage {stage_name}")
        pretrained = candidate
    elif 'pretrained' in model_cfg and model_cfg['pretrained']:
        pretrained = model_cfg['pretrained']
    else:
        pretrained = None
        
    if pretrained is not None:
        print(f"load form {pretrained}")
        model = YOLO(pretrained)
    else:
        model = YOLO(model_yaml) if model_yaml else None    

    # build train kwargs: merge top-level commonly used args and stage args
    train_kwargs = {}
    # top-level defaults
    if 'data' in top_cfg:
        train_kwargs['data'] = top_cfg['data']
    # ensure project/name passed
    train_kwargs['project'] = top_cfg.get('project', 'runs')
    train_kwargs['name'] = top_cfg.get('name', 'exp')
    train_kwargs['imgsz'] = top_cfg.get('imgsz', 640)

    # copy stage options except control keys
    for k, v in stage_cfg.items():
        if k in ('name', 'enabled', 'pretrained'):
            continue
        # map img_size key name if present
        else:
            train_kwargs[k] = v

    print(f"Train kwargs for {stage_name}: { {k:v for k,v in train_kwargs.items() if k not in ('project','name','data')} }")
    results = model.train(**train_kwargs)
    
    return model, results

def main(config_path):
    cfg = load_yaml(config_path)
    project, name = prepare_env(cfg)
    # set CUDA env if present at top-level
    if cfg.get('cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_visible_devices'])

    model_cfg = cfg.get('model', {})
    # model = load_model(model_cfg)

    stages = cfg.get('stages', [])
    for idx, stage in enumerate(stages):
        if not stage.get('enabled', True):
            print(f"ignore stage {stage.get('name', idx)} (enabled=false)")
            continue

        # 如果不是第一个启用阶段，且没有在 stage 指定 pretrained，则尝试加载上一次训练的 last.pt
        # candidate = os.path.join(cfg.get('project','runs'), cfg.get('name','exp'), 'weights', 'last.pt')


        model, results = run_stage(model_cfg, stage, cfg)

    print("all done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='trainingProfile/glassBoard.yaml', help='训练配置 YAML 路径')
    args = parser.parse_args()
    main(args.config)