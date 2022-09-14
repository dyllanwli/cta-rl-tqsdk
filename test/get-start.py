from tqsdk import TqApi, TqAuth


api = TqApi(auth=TqAuth("ohoiohasd", "XeGGefv&3@p#HMxi"))
quote = api.get_quote("DCE.i2301")
print (quote.last_price, quote.volume)


while True:
    api.wait_update()
    print (quote.datetime, quote.last_price)