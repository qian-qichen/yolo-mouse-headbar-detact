import argparse, copy, yaml, os
def load_cli_args(defaults, helps):
    # load config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument( '-c','--cliconfig', default=None, help='path to yaml config file')
    pre_args, _ = pre_parser.parse_known_args()
    # override 
    merged = copy.deepcopy(defaults)
    if pre_args.cliconfig:
        if not os.path.exists(pre_args.cliconfig):
            raise FileNotFoundError(f"Config file not found: {pre_args.cliconfig}")
        with open(pre_args.cliconfig, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            merged[k] = v
    # build former strick parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cliconfig', '-c', default=pre_args.cliconfig, help='path to yaml config file')

    def bool_cast(s):
        if isinstance(s, bool):
            return s
        return str(s).lower() in ('true', 'yes', 'y', 't')

    for key, default in merged.items():
        help_text = helps.get(key, '') if isinstance(helps, dict) else ''
        arg_name = f'--{key}'
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=bool_cast, default=default, help=help_text)
        elif isinstance(default, (dict, list)):
            parser.add_argument(arg_name, type=lambda s: yaml.safe_load(s), default=default, help=help_text)
        elif default is None:
            parser.add_argument(arg_name, default=None, help=help_text)
        else:
            parser.add_argument(arg_name, type=type(default), default=default, help=help_text)
    # get cli and delete cilconfig path
    parsed = parser.parse_args()
    out = vars(parsed)
    out.pop('cliconfig')
    return parsed,out