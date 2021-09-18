import os
import sys


def toolDelete(paths, begin, end):
    for path in paths:
        for folder_name, sub_folders, file_names in os.walk(path):
            for file_name in file_names:
                pre, post = file_name.split('.')
                pre_index = int(pre)
                if begin <= pre_index <= end:
                    os.remove(os.path.join(folder_name, file_name))
                    print("{} deleted".format(folder_name + file_name))


if __name__ == "__main__":
    begin_index = sys.argv[1]
    end_index = sys.argv[2]
    cur_path = os.getcwd()
    relative_img_path = "JPEGImages"
    relative_anno_path = "Annotations"
    paths = [os.path.join(cur_path, relative_img_path), os.path.join(cur_path, relative_anno_path)]
    print(paths)
    toolDelete(paths, int(begin_index), int(end_index))
