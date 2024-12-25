
1. cd /Users/ragheb/myprojects/stock/src
2. source ~/.bash_profile

# change the data config accordingly
    - /Users/ragheb/myprojects/stock/src/config/data_conf.json
        - monthly amo
        - weekly 1wk
        - minute 1m, daily 1d
    - In the data/driver.py you may want to make sure what is running is what you want
    - python3 data/driver.py
# set the training params
    - /Users/ragheb/myprojects/stock/src/config/train_conf.py
# run the data driver
    - python3 data/driver.py
# run the main driver xbg (and feature eng)
    - python3 driver.py
# run lstm
    - cd model/lstm
    - make sure you are reading the right file
    - python3 lstm.py


# github
1. generate a key locally
    1.1. ssh-keygen -t rsa -b 4096 -C "mahessaragheb@gmail.com"
    1.2.  eval "$(ssh-agent -s)"
    1.3.  ssh-add ~/.ssh/id_rsa
    1.4. cat ~/.ssh/id_rsa.pub
2. update or add the key in github settings
3. ssh -T git@github.com
