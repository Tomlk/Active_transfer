import os

def start_test(ratio,epoch_index,s_t_ratio):
    
    #不能用 os.system 会有并发问题

    import eval.test_SW_ICR_CCR as TEST

    net = "vgg16"
    part = "test_t"
    start_epoch = 6
    max_epochs = 12
    output_dir = "./data/experiments/SW_Faster_ICR_CCR/cityscapefoggy/result"
    dataset = "cityscapefoggy"

    models=os.listdir("./data/experiments/SW_Faster_ICR_CCR/cityscapefoggy/model/")
    modelfiles=models
    for item in modelfiles:
        if not item.endswith(".pth"):
            modelfiles.remove(item)

    modelfiles.sort()
    modelfiles.sort(key = lambda i:len(i),reverse=False) 
    print(modelfiles)
    currentmodel=modelfiles[-1]
    item=currentmodel.split('.')[0]
    modelepoch=item.split('_')[-1]

    GPUID=0

    print("modelepoch:",modelepoch)

    model_dir="./data/experiments/SW_Faster_ICR_CCR/cityscapefoggy/model/"+currentmodel


    
    print("ratio:",ratio)
    print("model:",currentmodel)
    result=TEST.excute(_GPUID=GPUID,_cuda=True,_gc=True,_lc=True,_part=part,_dataset=dataset,_model_dir=model_dir,_output_dir=output_dir,
                    _modelepoch=modelepoch,_ratio=ratio,_epochindex=epoch_index,_st_ratio=s_t_ratio)

    return result


    # command="CUDA_VISIBLE_DEVICES={} python eval/test_SW_ICR_CCR.py --cuda --gc --lc --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {} --ratio {} --epoch_index {} --st_ratio {}".format(
    #     GPUID,part, net, dataset, model_dir, output_dir, modelepoch,ratio,epoch_index,s_t_ratio
    # )
    # os.system(command)


# if __name__=="__main__":
#     start_test(ratio,epoch_index)
