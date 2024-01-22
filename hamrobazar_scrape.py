from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
import pandas as pd
import numpy as np
from selenium.webdriver.common.keys import Keys


options = Options()
options.add_argument('--headless')

driver = Chrome(options=options)

url = "https://hamrobazaar.com/category/mobile-phones-accessories/0618E1EF-00AF-4EAC-8E07-4978A2C7BB5E"

driver.get(url)

product_ele = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'product-list')))


for product_element in product_ele:
# Find the product title using a more specific identifier
    info = product_element.find_element(By.CLASS_NAME, 'seller__address').text
    print(info)






