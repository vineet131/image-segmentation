from models import modded_unet, mobilenet_v2

def main(model_name, img_size, n_classes):
    model_dict = {'modded_unet':modded_unet.modded_unet(img_size, n_classes),
                  'mobilenet_v2':mobilenet_v2.mobilenet_v2(img_size, n_classes)}
    model_final = model_dict[model_name]
    print(model_final.summary())
    return model_final