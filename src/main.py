from API import API
from commodity import Commodity
from policies.policy import Policy

def main():
    tqAPI = API()
    api  = tqAPI.api
    auth = tqAPI.auth
    cmod = Commodity()
    symbol = cmod.getCommodityConfig('铁矿石') + "2301"

    policy = Policy()

    policy.load_policy('random_forest')(auth, symbol)

    # quote = api.get_quote(symbol)
    # while True:
    #     api.wait_update()
    #     print(quote)




if __name__ == "__main__":
    main()
