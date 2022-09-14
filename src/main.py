from API import API
from commodity import Commodity


def main():
    api = API().api
    cmod = Commodity()

    symbol = cmod.getCommodityConfig('铁矿石')
    quote = api.get_quote(symbol + "2301")
    while True:
        api.wait_update()
        print(quote)




if __name__ == "__main__":
    main()
