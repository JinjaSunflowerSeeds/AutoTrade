import os

"""
cmd:
    cd /Users/ragheb/myprojects/stock/src
    python3 data/driver.py > files/logs/terminal/driver.txt 2>&1 
"""
if __name__ == "__main__":
    os.system("python3 data/ohlcv.py")
    os.system("python3 data/economy.py")
    os.system("python3 data/merger.py")
