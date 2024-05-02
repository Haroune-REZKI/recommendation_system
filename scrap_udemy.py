from selenium import webdriver
import re
import time

def get_rank(webpage, outputfile):
    """
    Purpose: obtain COURSE list from the input web page.
    :param webpage: the webpage that has products.
    :param outputfile: output file that has the product information.
    """
    count = 1
    page = 1
    f = open(outputfile, 'wt')
    f.write('id\tname\tlink\tseller\tprice\toriginal_price\tlectures\thours\tlevel\t' +
            time.strftime("%c") + '\n')
    # driver = webdriver.Chrome()
    driver = webdriver.Firefox()

    while count < 1000:
        if count == 1:
            tem_webpage = webpage
        else:
            page += 1
            tem_webpage = webpage + "?p=" + str(page)

        driver.get(tem_webpage)
        time.sleep(30)

        # parse the elements in the webpage.
        xpath_product = "//div[@class='popper--popper--19faV popper--popper-hover--4YJ5J']"
        product = driver.find_elements_by_xpath(xpath_product)

        for i in range(len(product)):
            try:
                link = product[i].find_element_by_class_name("browse-course-card--link--3KIkQ").get_attribute('href')
                name = link.replace('https://www.udemy.com/course/', '')
            except:
                continue

            try:
                tem_seller = product[i].find_element_by_class_name(
                    "course-card--instructor-list--lIA4f").text
                seller = re.sub('[^A-Za-z0-9_]', ' ', tem_seller)
            except:
                seller = "NULL"

            try:
                tem_price = product[i].find_element_by_class_name(
                    "course-card--discount-price--3TaBk").text
                price = tem_price.replace("$", "").replace("Current price", "").replace("\n", "")
            except:
                price = 'NULL'

            try:
                tem_original_price = product[i].find_element_by_class_name(
                    "price-text--original-price--2e-F5").text
                original_price = tem_original_price.replace("$", "").replace("Original Price", "").replace("\n","")
            except:
                original_price = 'NULL'

            try:
                other_info = product[i].find_element_by_class_name("course-card--course-meta-info--1hHb3").text
                other_info = other_info.replace("\n", '').replace("lectures", ',').replace("total hours", ',')
                other_info = other_info.replace("lecture", ',').replace("hour", ',')
                other_info = other_info.split(',')
                lectures = other_info[1]
                hours = other_info[0]
                level = other_info[2]

            except:
                continue
                # other_info = 'NULL'

            f.write(str(count) + '\t' + name + '\t' + link + '\t' + seller + '\t' + price + '\t'
                    + original_price + '\t' + lectures + '\t' + hours + '\t' + level + '\n')

            count += 1

    driver.quit()
    f.close()


if __name__ == "__main__":
    webpage = "https://www.udemy.com/courses/"
    outputfile = "course_list.tsv"
    get_rank(webpage, outputfile)