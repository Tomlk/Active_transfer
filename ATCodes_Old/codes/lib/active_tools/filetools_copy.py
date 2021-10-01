import os
import random
import shutil
import cv2
import xml.etree.ElementTree as ET
import sys
sys.path.append("..")


def get_all_file_list(root_path, key=lambda x: int(x[:-4])):
    """
    返回当前目录下的所有文件列表
    :param root_path: 当前目录
    :param key: 指定的排序逻辑需截取的文件名的哪一部分的 lambda 函数
    :return:
    """
    file_list = []
    # key=lambda x: int(x[:-4]) : 倒着数第四位'.'为分界线，按照'.'左边的数字从小到大排序
    for f in sorted(os.listdir(os.path.join(root_path)), key=lambda x: int(x[:-4])):   # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(root_path, f)
        if os.path.isfile(sub_path):
            file_list.append(sub_path)
    return file_list


def text_save(filename, datas):
    """
    将数据写入 txt 文件
    :param filename: 为写入CSV文件的路径
    :param data: 为要写入数据列表
    :return:
    """
    file = open(filename, 'w')  # 参数 w 表示写（重新写），a 表示追加；
    for data in datas:
        file.write(data.__str__() + '\n')
    file.close()
    print("------保存文件成功------")


def rename(file_path):
    """
    将给定目录下的图像文件重新命名（0000.jpg 格式）
    :param file_path:
    :return:
    """
    # 首先得到所有的图片（list）
    images = []
    # key=lambda x: int(x[:-4]) : 倒着数第四位'.'为分界线，按照'.'左边的数字从小到大排序
    for f in sorted(os.listdir(os.path.join(file_path)), key=lambda x: int(x[4:-4])):  # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(file_path, f)
        if os.path.isfile(sub_path):
            images.append(sub_path)
    print(images)
    # 重新命名
    count = 0
    for img in images:
        if count < 10:
            os.rename(img, os.path.join(file_path, '000' + str(count) + ".jpg"))
        elif count < 100:
            os.rename(img, os.path.join(file_path, '00' + str(count) + ".jpg"))
        elif count < 1000:
            os.rename(img, os.path.join(file_path, '0' + str(count) + ".jpg"))
        count += 1
    print("rename completed!")


def save_img_detected(out_dir, boxes, names, scores, src_img, labels, min_thresh):
    # 若输出路径不存在，则创建
    # if os._exists(out_dir):

    # 将检测结果画到图像中，然后保存检测结果图像
    for idx in range(boxes.shape[0]):
        # 返回一堆的 bbox，但是仅筛选出超过指定阈值（args.score）的 bbox
        if scores[idx] >= min_thresh:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names[labels[idx].item()]
            score = str(round(scores[idx].item(), 3))  # round(2.3456, 3) = 2.345  即保留小数点后 3 位
            # 画上检测框
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            # 添上类别说明
            cv2.putText(src_img, text=name + ' ' + score, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))
    cv2.imwrite(filename=out_dir, img=src_img)  # 保存图片


def rename_from_a(file_path, file_type, from_=0, lengthof=4, sort='number'):
    """
    按指定的起始数字开始依次重命名文件
    Args:
        file_path: 存放所有文件的文件夹路径
        from_: 从该数字开始
        file_type: 文件格式
    :return:
    """
    # 首先得到所有的文件（list）
    files = []
    # key=lambda x: int(x[:-4]) : 倒着数第四位'.'为分界线，按照'.'左边的数字从小到大排序
    if sort == 'filename':
        # 直接依据文件名排序
        for f in sorted(os.listdir(os.path.join(file_path))):  # 获取的列表是乱序的，记得排一下序
            sub_path = os.path.join(file_path, f)
            if os.path.isfile(sub_path):
                files.append(sub_path)
    if sort == 'number':
        # 依据文件名中的数字排序
        for f in sorted(os.listdir(os.path.join(file_path)), key=lambda x: int(x[:-4])):  # 获取的列表是乱序的，记得排一下序
            sub_path = os.path.join(file_path, f)
            if os.path.isfile(sub_path):
                files.append(sub_path)
    print('---------------------- 看看是否按序排列了 ------------------------')
    for ik in files:
        print(ik)
    print("lens:", len(files))
    print('----------------------------------------------------------------')
    # 重新命名
    count = from_
    for file_ in files:
        filename = str(count).zfill(lengthof)
        rename_ = os.path.join(file_path,  filename + ".{}".format(file_type))
        if file_type == 'xml':  # 如果是 xml 文件重命名，还要修改文件中的 filename 属性
            doc = read_xml(file_)
            root = doc.getroot()
            sub1 = root.find('filename')
            # 修改标签内容
            # python 利用 os 库从文件路径中分离出文件名
            _, tempfilename = os.path.split(rename_)
            sub1.text = os.path.splitext(tempfilename)[0] + '.jpg'
            # sub1.text = filename + '.jpg'     # 写法二
            # 保存修改
            doc.write(file_)  # 保存修改
        os.rename(file_, rename_)  # 重命名
        count += 1
    print("rename_from_a completed!")


def read_xml(in_path):
    """
    读取并解析 xml 文件
    :param in_path:
    :return:
    """
    tree = ET.parse(in_path)
    return tree


def read_txt_file(file_path):
    """
    按行读取 txt 文件内容，返回一个包含每一行内容的 list
    Args:
        file_path: txt 文件路径
    Returns: 返回 list（存储每一行内容）
    """
    lists = []
    f = open(file_path, 'r')  # 读文件
    for line_ in f:
        # print(line_.rstrip())
        lists.append(line_.rstrip())  # 需要使用 rstrip() 取出行末的换行符
    f.close()
    return lists


def remove_imgs(file_path, to_file_path, file_struct='dict'):
    """将通过不确定性选出来的图片及其标注转移到一个临时文件夹中
    Args:
        file_path: 存储挑选出来的图片信息的文件
        to_file_path：将要转去的位置
    注意：需要在 file_path 下创建 annotations 和 images 两个文件夹
    """
    imgs = []       # 存储所有要转移的图片（位置）
    f = open(file_path, 'r')  # 读文件
    for line_ in f:
        print(line_.rstrip())
        if file_struct == 'dict':   # 若文件中每一行都是 dict 形式的存储形式，需要取出图片路径属性
            # 将读取的一行 str 转成 dict 形式
            img_info = eval(line_)
            img_path = img_info['img_path']
            # for key, value in img_info.items():
            #     print(key, value)
            imgs.append(img_path)
        else:   # 若文件中每一行都是 str 形式，其本身就是图片路径
            imgs.append(line_.rstrip())     # 需要使用 rstrip() 取出行末的换行符
    f.close()

    # 转移图片和标注位置
    for img in imgs:
        # 获得标注文件的地址
        # 'img_path': 'E:\\AllDateSets\\MilitaryOB_5_class_torch\\unlabel_pool\\images\\0453.jpg'
        ss = os.path.dirname(img)
        ss = os.path.dirname(ss)
        ano_file = os.path.join(ss, 'Annotations', img.split(os.sep)[-1].split('.')[0] + '.xml')
        # 先判断目标文件夹是否存在
        if not os.path.exists(os.path.join(to_file_path, 'images')):
            ex = Exception("图片文件夹不存在")  # 创建异常对象
            raise ex    # 抛出异常对象
        if not os.path.exists(os.path.join(to_file_path, 'annotations')):
            ex2 = Exception("注释文件夹不存在")  # 创建异常对象
            raise ex2    # 抛出异常对象
        # 转移图片文件
        shutil.move(img, os.path.join(to_file_path, 'images'))
        # 转移标注信息文件
        shutil.move(ano_file, os.path.join(to_file_path, 'annotations'))
    print("----转移成功----")


def random_list_generator(start, stop, length):
    """
    返回不包含重复数字的随机数列表，产生的随机数字在 [start, stop) 之间，其个数为 length
    :param start: 起始值（可能包含在产生的列表内）
    :param stop: 终止值（不会包含在产生的列表内）
    :param length: 产生的列表的长度
    :return:
    """
    if length > stop - start + 1:
        print("长度不符合要求")
        return
    random_list = []
    count = 0
    while True:
        # randint(low, high):返回随机的整数，位于半开区间[low, high)
        random_num = random.randint(start, stop)
        if not random_list.__contains__(random_num):
            random_list.append(random_num)
            count += 1
        if count >= length:
            break
    return random_list


def filterimg():
    """
    acl-tl 部分
    筛掉重复出现在 test、train、unlabelpool 中的图片，只返回从未出现过的图片，并将其加入到 unlabelpool 中
    """
    file_path = r'/home/jiangb/notdetectedand14detected.txt'
    imgs_root = r'/home/jiangb/D2_acl_tl/watercolor_ac'
    right_ret_imgs = read_txt_file(file_path)
    # print(right_ret_imgs, len(right_ret_imgs))
    print('right_ret_imgs:', len(right_ret_imgs))
    filtered_imgs = []
    test_imgs = [img.split(os.sep)[-1].split('.')[0] for img in get_all_file_list(os.path.join(imgs_root, 'test', 'JPEGImages'))]
    train_imgs = [img.split(os.sep)[-1].split('.')[0] for img in get_all_file_list(os.path.join(imgs_root, 'train', 'JPEGImages'))]
    unlabelpool_imgs = [img.split(os.sep)[-1].split('.')[0] for img in get_all_file_list(os.path.join(imgs_root, 'unlabelpool', 'JPEGImages'))]
    print('test_imgs:', len(test_imgs))
    print('train_imgs:', len(train_imgs))
    print('unlabelpool_imgs:', len(unlabelpool_imgs))
    rep_in_train = []
    rep_in_test = []
    rep_in_unlabelpool = []
    for img in right_ret_imgs:
        if img in test_imgs:
            rep_in_test.append(img)
            continue
        elif img in train_imgs:
            rep_in_train.append(img)
            continue
        elif img in unlabelpool_imgs:
            rep_in_unlabelpool.append(img)
            continue
        else:
            filtered_imgs.append(img)
    print('---filtered_imgs---:', len(filtered_imgs))
    print('rep_in_test:', len(rep_in_test))
    print('rep_in_train:', len(rep_in_train))
    print('rep_in_unlabelpool:', len(rep_in_unlabelpool))

    # 将未重复的图片转移到 unlabelpool 下，训练后返回


def object_check():
    xml_path = r'E:\AllDateSets\ATL_data_checkup\cityspace\rename\source\Annotations'
    xmls = get_all_file_list(xml_path)
    for xml in xmls:
        doc = read_xml(xml)
        root = doc.getroot()
        obj = root.findall('object')
        if len(obj) == 0:
            print(xml)
            img = os.path.join(os.path.dirname(os.path.dirname(xml)), "JPEGImages", xml.split(os.sep)[-1].split('.')[0] + ".jpg")
            print(img)
            os.remove(xml)
            os.remove(img)


if __name__ == '__main__':
    pass

