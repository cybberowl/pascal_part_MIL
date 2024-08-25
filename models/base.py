def get_config_from_locals(**kwargs):
    config = kwargs
    config.pop('self')
    return config