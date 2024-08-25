def get_config_from_locals(**kwargs):
    config = kwargs
    config.pop('self')
    for key in config:
        if hasattr(config[key],'__name__'):
            config[key] = config[key].__name__
    return config