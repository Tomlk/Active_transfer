import os
import random
from shutil import copyfile
import sys  #


def gettxt(datapath,train_test_ratio):
    print(datapath)
    imgs1=os.listdir(os.path.join(datapath,"JPEGImages"))
    annos1=os.listdir(os.path.join(datapath,"Annotations"))
    imgs=imgs1
    annos=annos1
    for item in imgs1:
        if not item.endswith(".jpg"):
            imgs.remove(item)
    
    for item in annos1:
        if not item.endswith(".xml"):
            annos.remove(item)
    
    assert len(imgs)==len(annos)

    random.shuffle(imgs)
    ratio=train_test_ratio
    trainnums=int(ratio*len(imgs))
    trainfiles=imgs[0:trainnums]
    testfiles=imgs[trainnums:]
    trainfiles.sort()
    testfiles.sort()

    with open(os.path.join(datapath,"ImageSets","Main","trainval.txt"),"w") as f:
        s=""
        for item in trainfiles:
            s+=item.split('.')[0]
            s+="\n"
        f.write(s)
    
    copyfile(os.path.join(datapath,"ImageSets","Main","trainval.txt"),os.path.join(datapath,"ImageSets","Main","train.txt"))

    if ratio==1:
        copyfile(os.path.join(datapath,"ImageSets","Main","trainval.txt"),os.path.join(datapath,"ImageSets","Main","test.txt"))
    else:
        with open(os.path.join(datapath,"ImageSets","Main","test.txt"),"w")  as f:
            s=""
            for item in testfiles:
                s+=item.split('.')[0]
                s+="\n"
            f.write(s)

if __name__=="__main__":
    pass