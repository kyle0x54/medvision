import argparse


def load_config_cmdline(load_default_config,
                        update_config,
                        description='',
                        log_cfg=False):
    """ Load configurations from command line.

    Args:
        load_default_config (function): function to load default
            configurations.
        update_config (function): function to update secondary
            configurations according to primary configurations.
        description(str): refer to argparse.ArgumentParser description.
        log_cfg(bool): whether or not to log configurations.

    Return:
        (yacs CfgNode): loaded configurations

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
    cfg = load_default_config()

    # 3 load configuration from command line
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # 4. update configurations if primary configuration changes
    update_config(cfg)

    cfg.freeze()

    # 5. log configuration for visual check
    if log_cfg:
        print('Running with config:\n{}'.format(cfg))

    return cfg


def load_config(load_default_config,
                update_config,
                config_file_path=None):
    """ Load configuration.

    Args:
        load_default_config (function): function to load default
            configurations.
        update_config (function): function to update secondary
            configurations according to primary configurations.
        config_file_path (str): configuration file path.

    Return:
        (yacs CfgNode): loaded configurations
    """
    # TODO: add unit test
    cfg = load_default_config()

    if config_file_path is not None:
        cfg.merge_from_file(config_file_path)
    update_config(cfg)

    cfg.freeze()
    return cfg
