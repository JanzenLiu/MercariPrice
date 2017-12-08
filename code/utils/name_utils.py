STAT_NAMES = ['mean', 'std', 'min', 'p25', 'p50', 'p75', 'max']


def add_prefix(names, prefix, prefix_sep='_'):
    return [prefix_sep.join([prefix, name]) for name in names]


def get_stat_names(prefix, prefix_sep='_'):
    return add_prefix(STAT_NAMES, prefix, prefix_sep)
