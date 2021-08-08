import sys
from nepaldatascrapper import NEPSE

nepse = NEPSE()

def nepseSharePrice(symbol,date=""):
    return nepse.getSharePrice(symbol,date)

if __name__ == "__main__":
    print(nepseSharePrice("BPCL","2020-07-01"))