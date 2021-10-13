import os

def get_current_model(model_dir):
    models = os.listdir(model_dir)
    model_files = models
    for item in model_files:
        if not item.endswith(".pth"):
            model_files.remove(item)
    model_files.sort()
    model_files.sort(key=lambda i: len(i), reverse=False)
    current_model = model_files[-1]
    model_epoch=current_model.split('_')[-1]
    return current_model,model_epoch
