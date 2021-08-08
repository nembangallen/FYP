import random
def updated_forecast(n,low,high):
    print('n: ', n)
    print('low: ', low)
    print('high: ', high)
    result = []
    random.seed(10)
    for x in range(n):
        result.append(random.randint(low,high))
    return result

def nhcp_result():
    random.seed(10)
    return random.randint(-100,250)
