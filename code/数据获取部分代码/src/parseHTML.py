# 该代码用来解析从蝉妈妈网站上爬取下来的html
# 从中提取出视频名称、达主名称、点赞数、转发数、评论数、预估销量、预估销售额
# 并存入excel文件中
import codecs
from lxml import etree
import pandas as pd
import numpy as np

# 去除数字中间的','
# 解决有的数字右面带w(万)
def str_to_num(str:str)->float:
    str = str.replace(',', '')
    res = 0
    if str[-1] == 'w':
        res = float(str[:-1])*10000
    else:
        res = float(str)
    return res

def get_info(filename):
    # 打开html并解析成html tree
    f=codecs.open(filename,"r","utf-8")
    content=f.read()
    f.close()
    html=etree.HTML(content)

    # 各种参数值
    video_name = []
    up_name = []
    video_link = []
    like = [] #点赞
    forward = [] #转发
    comment = [] #评论
    sales_number = [] #销量
    sales_money = [] #销售额

    # 处理视频名称、up名称和视频链接
    a_tags = html.xpath('//tbody//a')
    i = 0
    for a in a_tags:
        if i % 6 == 1:
            # print('video_name: ', a.text)
            video_name.append(a.text)
        elif i % 6 == 3:
            # print('up_name: ', a.text)
            up_name.append(a.text)
        elif i % 6 == 5:
            # print('video_link: ', a.attrib['href'])
            video_link.append(a.attrib['href'])
        i += 1

    # 处理点赞、转发、评论、预估销量、预估销售额
    td_tags = html.xpath('//tbody//td')
    i = 0
    for td in td_tags:
        if i % 9 == 3:
            # print('点赞： ', td.text)
            like.append(str_to_num(td.text))
        elif i % 9 == 4:
            # print('转发 ', td.text)
            forward.append(str_to_num(td.text))
        elif i % 9 == 5:
            # print('评论 ', td.text)
            comment.append(str_to_num(td.text))
        elif i % 9 == 6:
            # print('预估销量 ', td.text)
            sales_number.append(str_to_num(td.text))
        elif i % 9 == 7:
            # print('预估销售额 ', td.text)
            sales_money.append(str_to_num(td.text))
            # print('---------------------------')
        i = i + 1

    info = []

    for i in range(0,len(video_name)):
        line_info = []
        line_info.append(video_name[i])
        line_info.append(up_name[i])
        line_info.append(video_link[i])
        line_info.append(like[i])
        line_info.append(forward[i])
        line_info.append(comment[i])
        line_info.append(sales_number[i])
        line_info.append(sales_money[i])
        info.append(line_info)
    return info

def store_to_excel(info, columns, filename):
    df = pd.DataFrame(info, columns=columns)
    df.to_excel(filename, index=False)

if __name__=='__main__':
    columns = ['视频名称', '达主', '视频链接', '点赞数', '转发数', '评论数', '预估销量', '预估销售额']
    for date in range(17,24):
        filename = '2020-07-' + str(date) + '.html'
        info = get_info(filename)
        store_to_excel(info,columns,'2020-07-'+str(date)+'.xlsx')
