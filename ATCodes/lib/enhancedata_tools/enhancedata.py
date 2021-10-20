# 数据增强
import os
import shutil
import cv2
import xml.etree.ElementTree as ET


def read_xml(in_path):
    tree = ET.parse(in_path)
    return tree


def flip(cvimg, type, rename_img, xml, rename_xml):
    img_samp = None
    # 修改 xml 文件标注位置
    doc = read_xml(xml)
    root = doc.getroot()
    root.find('filename').text = rename_img.split(os.sep)[-1]
    size = root.find('size')
    objs = root.findall('object')
    if type == 'horizontal':
        img_samp = cv2.flip(cvimg, 1, dst=None)
        width = int(size.find('width').text)
        for obj in objs:
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)

            tempxmin=min(xmin,xmax)
            tempxmax=max(xmin,xmax)

            xmin=tempxmin
            xmax=tempxmax

            new_xmin=max(0,width - xmax)
            new_xmax=min(width,width - xmin)
            bbox.find('xmin').text = str(new_xmin)
            bbox.find('xmax').text = str(new_xmax)
            if int(bbox.find('xmin').text) >int(bbox.find('xmax').text):
                print("file:",xml)
            if int(bbox.find('ymin').text) > int(bbox.find('ymax').text):
                print("file:",xml)
            assert int(bbox.find('xmin').text) <= int(bbox.find('xmax').text)
            assert int(bbox.find('ymin').text) <= int(bbox.find('ymax').text)
    if type == 'vertical':
        img_samp = cv2.flip(cvimg, 0, dst=None)
        height = int(size.find('height').text)
        for obj in objs:
            bbox = obj.find('bndbox')
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            tempymin=min(ymin,ymax)
            tempymax=max(ymin,ymax)

            ymin=tempymin
            ymax=tempymax

            new_ymin=max(0,height - ymax)
            new_ymax=min(height,height - ymin)
            bbox.find('ymin').text = str(new_ymin)
            bbox.find('ymax').text = str(new_ymax)

            if int(bbox.find('xmin').text) >int(bbox.find('xmax').text):
                print("file:",xml)
            
            if int(bbox.find('ymin').text) > int(bbox.find('ymax').text):
                print("file:",xml)

            assert int(bbox.find('xmin').text) <= int(bbox.find('xmax').text)
            assert int(bbox.find('ymin').text) <= int(bbox.find('ymax').text)
    cv2.imwrite(rename_img, img_samp)
    doc.write(rename_xml)
    # cv2.imshow('img_' + type + '_flip', img_samp)


def up_down_sample(cvimg, rename_img, type, xml, rename_xml):
    img_samp = None
    if type == 'up':
        img_samp = cv2.pyrUp(cvimg)
    if type == 'down':
        img_samp = cv2.pyrDown(cvimg)
    # cv2.imshow("img_samp" + type, img_samp)
    cv2.imwrite(rename_img, img_samp)
    # 调整 xml
    h, w, _ = img_samp.shape
    doc = read_xml(xml)
    root = doc.getroot()
    root.find('filename').text = rename_img.split(os.sep)[-1]
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    size.find('width').text = str(w)
    size.find('height').text = str(h)
    objs = root.findall('object')
    for obj in objs:
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        new_xmin = int((w / width) * xmin)
        if new_xmin > w:
            new_xmin = w
        new_xmax = int((w / width) * xmax)
        if new_xmax > w:
            new_xmax = w
        new_ymin = int((h / height) * ymin)
        if new_ymin > h:
            new_ymin = h
        new_ymax = int((h / height) * ymax)
        if new_ymax > h:
            new_ymax = h
        bbox.find('xmin').text = str(new_xmin)
        bbox.find('xmax').text = str(new_xmax)
        bbox.find('ymin').text = str(new_ymin)
        bbox.find('ymax').text = str(new_ymax)
    doc.write(rename_xml)


def data_enhance(img, xml, type, addcharacter, save_path):
    """
    数据增强.
    Args:
        img: 传入图片
        xml: 传入标注
        type: 增强类型 {0:直接复制, 1:水平翻转, 2:垂直翻转, 3:上采样, 4:下采样}
        addcharacter: 重命名文件时，文件名前添加字符
        save_path: 新文件（img、xml）保存路径
    Returns:
        None
    """

    JPEGImages_path=os.path.join(save_path,"JPEGImages")
    Annotations_path=os.path.join(save_path,"Annotations")
    rename_img = os.path.join(JPEGImages_path, addcharacter + img.split(os.sep)[-1])
    rename_xml = os.path.join(Annotations_path, addcharacter + xml.split(os.sep)[-1])
    img1 = cv2.imread(img)
    # cv2.imshow('initial image', img1)
    if type == 0:     # simply copy
        shutil.copy(img, rename_img)
        doc = read_xml(xml)
        root = doc.getroot()
        root.find('filename').text = rename_img.split(os.sep)[-1]
        doc.write(rename_xml)
    if type == 1:     # horizontal flip
        flip(img1, type='horizontal', rename_img=rename_img, xml=xml, rename_xml=rename_xml)
    if type == 2:      # vertical flip
        flip(img1, type='vertical', rename_img=rename_img, xml=xml, rename_xml=rename_xml)
    if type == 3:       # up sample
        up_down_sample(img1, rename_img=rename_img, type='up', xml=xml, rename_xml=rename_xml)
    if type == 4:       # down sample
        up_down_sample(img1, rename_img=rename_img, type='down', xml=xml, rename_xml=rename_xml)
    # cv2.waitKey(10000)


if __name__ == '__main__':
    save_path = r'.\\output\\enhancedata'
    img_path = r'.\\output\\initialdata\\00000026.jpg'
    xml_path = r'.\\output\\initialdata\\00000026.xml'
    # print(save_path.split(os.sep))
    data_enhance(img=img_path, xml=xml_path, type=2, addcharacter='f', save_path=save_path)
