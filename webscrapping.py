from bs4 import BeautifulSoup
import requests
import pandas as pd


 
# page_num = 0
# rows=[]
# for i in range(1, 7):
#     page_num = page_num + 1
#     url = 'http://www.scrapethissite.com/pages/forms/?page_num=' + str(page_num) + '&per_page=100'
#     page = requests.get(url)

#     soup = BeautifulSoup(page.text, 'html.parser')

#     if i == 1:
#         title = soup.find_all('th')
#         title_names = [data.get_text(strip=True) for data in title]

#         df = pd.DataFrame(columns=title_names)

#     row_data = soup.find_all('td')

#     individeual_row_data = [data.get_text(strip=True) for data in row_data]

#     chunk_size = len(title_names)
#     for i in range(0, len(individeual_row_data), chunk_size):
#         row_data = individeual_row_data[i:i + chunk_size]
#         rows.append(row_data)

    
# df = pd.DataFrame(rows, columns = title_names)




# url = 'https://www.forbes.com/real-time-billionaires/#4b42941b3d78'

# page = requests.get(url)

# soup = BeautifulSoup(page.text, 'html.parser')

# print(soup.find('div',attrs = {'class':"scrolly-table"}))




# titles = soup.find_all()

# print(titles)

# title_name = [data.get_text for data in titles]















