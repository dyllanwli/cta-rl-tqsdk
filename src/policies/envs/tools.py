from commodity import Commodity




def get_symbol_by_name(name):
    cmod = Commodity()
    return cmod.get_instrument_name(name)