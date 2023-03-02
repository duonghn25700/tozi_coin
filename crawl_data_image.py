import bs4
import requests
from selenium import webdriver
import os
import time
from selenium.webdriver.common.by import By


'''
search_URL : Điền url hình ảnh vào 
name       : Tên thư mục muốn lưu files
folder_name: Đường dẫn đến thư mục lưu files
'''

# URL of image's tag
search_URL = "https://www.google.com/search?q=10+cents+australia+coin&sxsrf=AJOqlzVs5VfW4ZZfd465gZ7X7KwJBMnjCw:1677656012304&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjvqazVm7r9AhWVCd4KHaMQCg4Q_AUoAXoECAEQAw&biw=1536&bih=714&dpr=1.25"

# Your folder name
name = '4'
#######################################################################################################################


#creating a directory to save images
folder_name = r'D:\DevSenior_Training\coin_detection\new_update_data' + name
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, num):

    respond = requests.get(url)
    if respond.status_code == 200:
        with open(os.path.join(folder_name, str(num)+".jpg"), 'wb') as file:
            file.write(respond.content)

chromePath= r"D:\Download\chromedriver_win32\chromedriver.exe"
driver = webdriver.Chrome(chromePath)

driver.get(search_URL)

a = input("Waiting...")

driver.execute_script("window.scrollTo(0, 0);")

page_html = driver.page_source
pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')
containers = pageSoup.findAll('div', {'class': "isv-r PNCib MSM1fd BUooTd"})

print(len(containers))

len_containers = len(containers)

for i in range(1, len_containers+1):
    if i % 25 == 0:
        continue
    # //*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
    xPath = """//*[@id="islrg"]/div[1]/div[%s]"""%(i)
    previewImageXPath = """//*[@id="islrg"]/div[1]/div[%s]/a[1]/div[1]/img"""%(i)
    previewImageElement = driver.find_element(By.XPATH, previewImageXPath)
    previewImageURL = previewImageElement.get_attribute("src")

    driver.find_element(By.XPATH, xPath).click()

    timeStarted = time.time()

    while True:
    # //*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img
        imageElement = driver.find_element(By.XPATH, """//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div/div[1]/div[2]/div[2]/div/a/img""")
        imageURL= imageElement.get_attribute('src')

        if imageURL != previewImageURL:
            #print("actual URL", imageURL)
            break

        else:
            #making a timeout if the full res image can't be loaded
            currentTime = time.time()

            if currentTime - timeStarted > 10:
                print("Timeout! Will download a lower resolution image and move onto the next one")
                break
    #Downloading image
    try:
        download_image(imageURL, folder_name, i)
        print("Downloaded element %s out of %s total. URL: %s" % (i, len_containers + 1, imageURL))
    except:
        print("Couldn't download an image %s, continuing downloading the next one"%(i))