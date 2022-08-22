# import pdfplumber
# import pandas as pd
# def pdf_read():
#     pdf = pdfplumber.open("1-s2.0-S100200711260080X-main.pdf")
#     #pages=input("转换表格的页码")
#     p0=pdf.pages[4]
#     print(p0)
#     table=p0.extract_table()
#     print(table)
#     # df=pd.DataFrame(table[1:], columns=table[0])
#     # df.to_excel("bbbb.xlsx")
#
# if __name__ == '__main__':
#     pdf_read()



import tabula
import pprint

df1 = tabula.read_pdf('test.pdf', pages="5,6,7", encoding='Ansi', stream=True)

print(len(df1))
# print(df1[3])
df = df1[0]
df = df.append([df1[1], df1[2]])
pprint.pprint(len(df))
pprint.pprint(df1[0])

# df1[0].to_csv('test1.csv')
# df1[1].to_csv('test2.csv')
# df1[2].to_csv('test3.csv')