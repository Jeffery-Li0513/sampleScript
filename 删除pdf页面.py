'''
删除PDF文件的指定页
'''

from PyPDF2 import PdfFileReader, PdfFileWriter

readfile = r"E:\桌面\临时文献\22.pdf"
outfile = r"E:\桌面\临时文献\上传\22.pdf"

pdfReader = PdfFileReader(open(readfile, 'rb'))
pdfFileWriter = PdfFileWriter()
numPages = pdfReader.getNumPages()
pagelist=[0,5]   #注意第一页的index为0.
for index in range(0, numPages):
    if index in pagelist:
        pageObj = pdfReader.getPage(index)
        pdfFileWriter.addPage(pageObj)
pdfFileWriter.write(open(outfile, 'wb'))