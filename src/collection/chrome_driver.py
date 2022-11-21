import sys
import time

from selenium import webdriver
from selenium.common.exceptions import TimeoutException

start = time.time()
INTERVAL_TIME = 1  # Interval time between queries
urls = []
ct = 0

fname = "/vagrant/short_list_1500"
with open(fname) as f:
    lines = f.readlines()
    for line in lines:
        urls.append(line.strip())

url = urls[int(sys.argv[1])]
print(url)
print("Started display")
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome('/usr/local/bin/chromedriver', chrome_options=options)
driver.set_page_load_timeout(30)
print("Started driver")
url = 'http://' + url
try:
    driver.get(url)
    time.sleep(INTERVAL_TIME * 5)
except TimeoutException as ex:
    print(ex)
driver.quit()
stop = time.time()
print("Time taken:" + str(stop - start))
time.sleep(INTERVAL_TIME)
