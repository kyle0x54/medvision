import argparse


def load_config_cmdline(load_default, description='', log_cfg=False):
    """ Load configuration from command line.

    Args:
        load_default (yacs CfgNode): function to load default configuration.
        description(str): refer to argparse.ArgumentParser description.
        log_cfg(bool): whether or not to log configuration.

    Return:
        (yacs CfgNode): loaded configuration

    Example:
        <<< python train.py --config_file config.yaml
    """
    # TODO: add unit test
    # 1. define argument parser
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--config_file", default=None, type=str,
        help="path to yaml config file"
    )
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER,
        help="override config options using command-line",
    )

    args = parser.parse_args()

    # 2 load default configuration
    cfg = load_default()

    # 3 load configuration from command line
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # 4. log configuration for visual check
    if log_cfg:
        print('Running with config:\n{}'.format(cfg))

    return cfg


def load_config(load_default, config_file_path=None):
    """ Load configuration.

    Args:
        load_default (yacs CfgNode): function to load default configuration.
        config_file_path (str): configuration file path.

    Return:
        (yacs CfgNode): loaded configuration
    """
    # TODO: add unit test
    cfg = load_default()
    if config_file_path is not None:
        cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg
