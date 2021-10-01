import os

def start_test(ratio,epoch_index,s_t_ratio,dataset_name,GPUID):
    
    #不能用 os.system 会有并发问题

    import eval.test_SW_ICR_CCR as TEST

    net = "vgg16"
    part = "test_t"
    output_dir=os.path.join("./data/experiments/SW_Faster_ICR_CCR",dataset_name,"result")
    dataset = dataset_name

    models=os.listdir(os.path.join("./data/experiments/SW_Faster_ICR_CCR",dataset_name,"model"))

    # models=os.listdir("./data/experiments/SW_Faster_ICR_CCR/bddnight10/model/")
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

    # GPUID=1

    print("modelepoch:",modelepoch)

    model_dir=os.path.join("./data/experiments/SW_Faster_ICR_CCR",dataset_name,"model",currentmodel)


    
    print("ratio:",ratio)
    print("model:",currentmodel)
    result=TEST.excute(_GPUID=GPUID,_cuda=True,_gc=True,_lc=True,_part=part,_dataset=dataset,_model_dir=model_dir,_output_dir=output_dir,
                    _modelepoch=modelepoch,_ratio=ratio,_epochindex=epoch_index,_st_ratio=s_t_ratio,_test_flag=False)

    return result
