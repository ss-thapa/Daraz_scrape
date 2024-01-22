
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



url = "https://www.daraz.com.np/mens-t-shirts/?from=sideFilters&location=-168&page=1"

driver.get(url)

try:
    # Wait for the product element to be present
    product_ele = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'gridItem--Yd0sa')))
except Exception as e:
    print(f"An error occurred: {e}")
    driver.quit()

# Find the product title using a more specific identifier
df_data = []

for product_element in product_ele:
    # Find the product title using a more specific identifier
    product_name = product_element.find_element(By.CLASS_NAME, 'title-wrapper--IaQ0m').text


    
    original_price = product_element.find_element(By.CLASS_NAME, 'original-price--lHYOH').text
    if not original_price:
        original_price='0'



    current_price = product_element.find_element(By.CLASS_NAME, 'current-price--Jklkc').text
    if not original_price:
        original_price='0'

    try:
        ratingss = product_element.find_element(By.CLASS_NAME, 'rating-wrapper--caEhB').text
    except NoSuchElementException as e:
        ratingss = '0'


    x=ratingss.split('\n')
    if(len(x) < 3 ):
        x.insert(2, '0')
    
    ratings = x[0]
    no_of_ratings = x[1]
    sold_num = x[-1]   

    df_data.append({'product_name':product_name,'ratings':ratings, 'no_of_ratings':no_of_ratings, 'sold_num':sold_num,'original_price':original_price,'current_price':current_price})       

    
df = pd.DataFrame(df_data)

# Quit the driver
driver.quit()











