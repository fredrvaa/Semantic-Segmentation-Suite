import os,cv2,sys,math,time
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--folder', type=str, default=None, required=False, help='The folder you want to predict on')
parser.add_argument('--video', type=str, default=None, required=False, help='The video you want to predict on')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default='FC-DenseNet56', required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()


assert args.image or args.folder or args.video, "Argument --image, --folder or --video is required for prediciting"

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


if args.image:
    print("Testing image " + args.image)

    loaded_image = utils.load_image(args.image)
    orig_image = cv2.cvtColor(loaded_image, cv2.COLOR_RGB2BGR)  
    resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

    output_image = sess.run(network,feed_dict={net_input:input_image})

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    mask = helpers.colour_code_segmentation(output_image, label_values)
    mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR)

    resized_mask = cv2.resize(mask, (loaded_image.shape[1], loaded_image.shape[0]))

    pred_fg = np.copy(resized_mask)
    pred_fg[np.where((pred_fg==[255,255,255]).all(axis=2))] = [255,0,0]
    pred_fg[np.where((pred_fg!=[255,0,0]).all(axis=2))] = [0,0,0]

    resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

    mask_inv = cv2.bitwise_not(resized_mask)    
    bg = cv2.bitwise_or(orig_image, orig_image, mask=mask_inv)

    final = cv2.bitwise_or(pred_fg, bg)

    file_name = utils.filepath_to_name(args.image)
    cv2.imwrite("pred_images/{}_pred.png".format(file_name), final)
    cv2.imwrite("pred_images/{}_pred_mask.png".format(file_name), resized_mask)

    print("")
    print("Finished!")
    print("Wrote image " + "%s_pred.png"%(file_name))

elif args.video:
    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    while(cap.isOpened()):
        start_time = time.time()
        ret, loaded_image = cap.read()
        if ret:
            resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))
            color_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)  
            input_image = np.expand_dims(np.float32(color_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

            output_image = sess.run(network,feed_dict={net_input:input_image})

            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)

            mask = helpers.colour_code_segmentation(output_image, label_values)
            mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR)

            resized_mask = cv2.resize(mask, (loaded_image.shape[1], loaded_image.shape[0]))

            pred_fg = np.copy(resized_mask)
            pred_fg[np.where((pred_fg==[255,255,255]).all(axis=2))] = [255,0,0]
            pred_fg[np.where((pred_fg!=[255,0,0]).all(axis=2))] = [0,0,0]

            resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

            mask_inv = cv2.bitwise_not(resized_mask)    
            bg = cv2.bitwise_or(loaded_image, loaded_image, mask=mask_inv)

            final = cv2.bitwise_or(pred_fg, bg)

            file_name = utils.filepath_to_name(args.video)
            cv2.imshow('pred', final)
            #cv2.imshow('original', loaded_image)
            #cv2.imshow('mask', resized_mask)
            print("FPS: ", 1.0 / (time.time() - start_time), end="\r") # FPS = 1 / time to process loop
            if cv2.waitKey(33) == ord('a'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

elif args.folder:

    for file in os.listdir(args.folder):
        image_path = os.path.join(args.folder, file)

        loaded_image = utils.load_image(image_path)
        orig_image = cv2.cvtColor(loaded_image, cv2.COLOR_RGB2BGR)  
        resized_image = cv2.resize(loaded_image, (args.crop_width, args.crop_height))
        input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

        output_image = sess.run(network,feed_dict={net_input:input_image})

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        mask = helpers.colour_code_segmentation(output_image, label_values)
        mask = cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR)

        resized_mask = cv2.resize(mask, (loaded_image.shape[1], loaded_image.shape[0]))

        pred_fg = np.copy(resized_mask)
        pred_fg[np.where((pred_fg==[255,255,255]).all(axis=2))] = [255,0,0]
        pred_fg[np.where((pred_fg!=[255,0,0]).all(axis=2))] = [0,0,0]

        resized_mask = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)

        mask_inv = cv2.bitwise_not(resized_mask)    
        bg = cv2.bitwise_or(orig_image, orig_image, mask=mask_inv)

        final = cv2.bitwise_or(pred_fg, bg)

        file_name = utils.filepath_to_name(image_path)

        cv2.imwrite("pred_images/{}_pred.png".format(file_name), final)
        cv2.imwrite("pred_images/{}_pred_mask.png".format(file_name), resized_mask)

        print("Wrote image " + "%s_pred.png"%(file_name))
    print("Finished!")
