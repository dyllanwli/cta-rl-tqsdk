from commodity import Commodity
from .constant import EnvConfig




def get_symbols_by_names(config: EnvConfig):
    cmod = Commodity()
    return [cmod.get_instrument_name(name) for name in config.symbols]