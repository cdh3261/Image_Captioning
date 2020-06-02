import os
import csv
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import PIL.Image as pilimg
from tqdm import tqdm
from sklearn.utils import shuffle
import time

# 이미지 불러오기
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    # 이미지를 정규화하여 -1에서 1사이의 픽셀을 포함.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def setting_captions():
    f = open('./datasets/captions.csv', 'r')
    annotations = f.read()
    f.close()

    annotations = annotations.split('\n')

    all_captions = []
    all_img_name_vector = []
    for annot in annotations[1:-1]:
        img, caption_num, caption = annot.split('|')
        while caption[-1] == ',':
                caption = caption[:-1]
        
        caption = '<start>'+caption+'<end>'
        if img[0] == '"':
            img = img[1:]
        
        full_coco_image_path = '.\\datasets\\images\\' + img

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # 랜덤하게 섞는다.
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state = 1)

    # 처음 3만개를 선택
    num_examples = 30000
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return train_captions, img_name_vector



def make_npy(img_name_vector, image_features_extract_model):
    
    encode_train = sorted(set(img_name_vector))
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
    


def do_tokenizer(train_captions,top_k):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, oov_token="<unk>", filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    train_seqs = tokenizer.texts_to_sequences(train_captions)

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # 최대 길이
    max_length = max(len(t) for t in train_seqs)
    print("complete tokenizer")

    return train_seqs, cap_vector, tokenizer, max_length

def load_pickle():
    with open('./pickle.p', 'rb') as file:
        tokenizer = pickle.load(file)
        return tokenizer

def load_split():
    with open('./split.p', 'rb') as file:
        split = pickle.load(file)
        return split

def save_pickle(train_seqs, cap_vector, tokenizer, max_length):
    with open('./pickle.p', 'wb') as file:
        pickle.dump([train_seqs, cap_vector, tokenizer, max_length], file)


def save_split(img_name_train, img_name_val, cap_train, cap_val):
    with open('./split.p', 'wb') as file:
        pickle.dump([img_name_train, img_name_val, cap_train, cap_val], file)


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def do_dataset(top_k, img_name_train, cap_train):
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset



def image_data_augmentation(path):
    im = pilimg.open(path)
    image_data = np.array(im)
    # img = load_img(image)
    # convert to numpy arry
    # data = img_to_array(img)
    #expand dimension to one sample
    samples = expand_dims(image_data, 0)
    #create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=[-200,200])
    #prepare iterator
    it = datagen.flow(samples, batch_size=1)
    #generate samples and plot
    for i in range(9):
        plt.subplot(330+1+i)
        batch = it.next()
        #convert to unsigned intergers for viewing
        image = batch[0].astype('uint8')
        plt.imshow(image)
    plt.show()




def pltshow(loss_plot):
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

