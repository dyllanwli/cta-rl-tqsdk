from tqsdk import TqApi

class Account:

    def __init__(self, api: TqApi):
        self.api = api
        self.account = api.get_account()
    
    def getPosition(self, symbol: str):
        return self.api.get_position(symbol)