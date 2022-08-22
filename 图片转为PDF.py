import os
import fitz
import glob

def pic2pdf(img_path, pdf_path, pdf_name):
    for img in sorted(glob.glob(img_path + pdf_name + '.jpg')):
        doc = fitz.open()
        imgdoc = fitz.open(img_path)
        pdfbytes = imgdoc.convertToPDF()
        imgpdf = fitz.open("pdf", pdfbytes)
        doc.insertPDF(imgpdf)
        doc.save(pdf_path + pdf_name + '.pdf')
        doc.close()
        print('结束')


if __name__ == '__main__':
    img_path = r'E:\桌面\2020级硕士研究生学业奖学金评定'
    pdf_path = r'E:\桌面\2020级硕士研究生学业奖学金评定'
    pic2pdf(img_path, pdf_path, pdf_name='q2签字')