from config import config
from data import preprocess
from utils import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from models import encoder, decoder
from tqdm import tqdm
import numpy as np
import os
import time
import PIL.Image as pilimg
from PIL import Image
import predict

# config 저장
# utils.save_config(config)

# 이미지 파일 로드
imageName = '36979.jpg'
path = '.\\datasets\\images\\' + imageName
# img, image_path = preprocess.load_image(path)
# print("complete image load")

# 캡션 관리
train_captions, img_name_vector = preprocess.setting_captions()

# .npy 만들기
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
# preprocess.make_npy(img_name_vector, image_features_extract_model)


# 토큰화 하기
top_k = 5000 # 5000개의 많이 나오는 단어
# train_seqs, cap_vector, tokenizer, max_length = preprocess.do_tokenizer(train_captions, top_k)

# pickle save (처음 한번만 실행)
# preprocess.save_pickle(train_seqs, cap_vector, tokenizer, max_length)
# print('complete pickle save')

# pickle load
# tokenizer = preprocess.load_pickle()
tokens = preprocess.load_pickle()
train_seqs, cap_vector, tokenizer, max_length = tokens[0], tokens[1], tokens[2], tokens[3]
# print('complete pickle load')
# print("load token: ", tokenizer)

# 데이터를 train, test로 나눔 (처음 한번만 실행)
# img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.2, random_state=0)
# preprocess.save_split(img_name_train, img_name_val, cap_train, cap_val)
split = preprocess.load_split()
img_name_train, img_name_val, cap_train, cap_val = split[0], split[1], split[2], split[3]
# print(len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))

# 데이터 셋
dataset = preprocess.do_dataset(top_k, img_name_train, cap_train)


# IMAGE AUGEMENTATION
preprocess.image_data_augmentation(path)


# endcode, decode
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train)
featrues_shape = 2048
attention_features_shape = 64
encoder = encoder.CNN_Encoder(embedding_dim)
decoder = decoder.RNN_Decoder(embedding_dim, units, vocab_size)


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def checkout():
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0

    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) 
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)
    
    return start_epoch, ckpt_manager

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(img_tensor, target, tokenizer):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def training(start_epoch, dataset, num_steps):
    loss_plot = []
    EPOCHS = 20 # 학습 횟수
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0
        for (batch, (img_tensor, target)) in enumerate(dataset):
            # print(batch,(img_tensor, target))
            batch_loss, t_loss = train_step(img_tensor, target, tokenizer)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                            total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    return loss_plot

# checkout
start_epoch, ckpt_manager = checkout()

# training
loss_plot = training(start_epoch, dataset, num_steps)

# loss plot show
preprocess.pltshow(loss_plot)


# captions on the validation set
# 랜덤한 이미지 하나 선택
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = predict.evaluate(image, tokenizer, max_length, attention_features_shape, image_features_extract_model, encoder, decoder)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
predict.plot_attention(image, result, attention_plot, real_caption, ' '.join(result))