

from google.colab import drive
drive.mount('/content/drive')

import mido
from mido import MidiFile, MidiTrack, Message
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten,Bidirectional,Input, Dense, Reshape, Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from glob import glob


songs=glob('/content/drive/My Drive/hip_hop/*/*.mid')
mid = []

for i in songs:
  print(i)
  mid.append(MidiFile(i))

sequence_length=200
epochs=5

def train(Discriminator, Generator, X_train, sequence_length, epochs):
	for _ in range(epochs):
	  real = np.ones((128, 1))
	  fake = np.zeros((128, 1))
	  idx = np.random.randint(0, X_train.shape[0], 128)
	  real_seqs = X_train[idx]
	  noise = np.random.random((128, sequence_length))
	  gen_seqs = generator.predict(noise)
	  d_loss_real = discriminator.train_on_batch(real_seqs, real)
	  d_loss_fake = discriminator.train_on_batch(gen_seqs, fake)
	  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
	  noise = np.random.random((128, sequence_length))
	  g_loss = combined.train_on_batch(noise, real)
	  return d_loss, g_loss




def build_discriminator():
      model = Sequential()
      model.add(LSTM(512, input_shape=(sequence_length,1), return_sequences=True))
      model.add(Bidirectional(LSTM(512)))
      model.add(Dense(512))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dense(256))
      model.add(LeakyReLU(alpha=0.2))
      model.add(Dense(1, activation='sigmoid'))
      model.summary()
      seq = Input(shape=(sequence_length,1))
      validity = model(seq)
      return Model(seq, validity)

def build_generator():
        model = Sequential()
        model.add(Dense(256, input_dim=sequence_length))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod((sequence_length,1)), activation='tanh'))
        model.add(Reshape((sequence_length,1)))
        model.summary()        
        noise = Input(shape=(sequence_length,))
        seq = model(noise)
        return Model(noise, seq)

def create_model(optimizer, sequence_length):
	discriminator = build_discriminator()
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	        # Build the generator
	generator = build_generator()
	        # The generator takes noise as input and generates note sequences
	z = Input(shape=(sequence_length,))
	generated_seq = generator(z)
	        # For the combined model we will only train the generator
	discriminator.trainable = False
	validity = discriminator(generated_seq)
	        # The combined model  (stacked generator and discriminator)
	        # Trains the generator to fool the discriminator
	combined = Model(z, validity)
	combined.compile(loss='binary_crossentropy', optimizer=optimizer)
	combined.summary()
	return discriminator, generator, combined



optimizer = Adam(0.0002, 0.5)



############################################################################################
# GAN for notes variable
############################################################################################

# Build and compile the discriminator



discriminator, generator, combined = create_model(optimizer, sequence_length)

notes = []

for i in mid:
  for msg in i:
    if not msg.is_meta:
      data = msg.bytes()
      if len(data) == 3 and data[0]<191:
        notes.append(data[1])

  

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.array(notes).reshape(-1,1))
notes = list(scaler.transform(np.array(notes).reshape(-1,1)))
notes = [list(note) for note in notes]

X_train = []

n_prev = sequence_length
for i in range(len(notes)-n_prev):
    X_train.append(notes[i:i+n_prev])
X_train = np.array(X_train)



d_loss,g_loss = train(discriminator, generator, combined, X_train, sequence_length, epochs)
print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (1, d_loss[0], 100*d_loss[1], g_loss))
start = np.random.randint(0, len(X_train)-1)
pattern = X_train[start]
print ('pattern.shape', pattern.shape)
noise = np.random.random((1, sequence_length))
predict_notes = generator.predict(noise)


######################## Prediction of notes variable #######################################
predict_notes = np.squeeze(predict_notes)
predict_notes = np.squeeze(scaler.inverse_transform(predict_notes.reshape(-1,1)))
predict_notes = [int(i) for i in predict_notes]

############################################################################################
# GAN for velocity variable
############################################################################################


discriminator2, generator2, combined2 = create_model(optimizer, sequence_length)


vel = []

for i in mid:
  for msg in i:
    if not msg.is_meta:
      data = msg.bytes()
      if len(data) == 3 and data[0]<191:
        vel.append(data[1])

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(np.array(vel).reshape(-1,1))
vel = list(scaler.transform(np.array(vel).reshape(-1,1)))


vel = [list(v) for v in vel]

X_train = []

n_prev = sequence_length
for i in range(len(vel)-n_prev):
    X_train.append(vel[i:i+n_prev])

X_train = np.array(X_train)


d_loss,g_loss = train(discriminator2, generator2, combined2, X_train, sequence_length, epochs)
print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (1, d_loss[0], 100*d_loss[1], g_loss))


######################## Prediction of velocity variable #######################################
start = np.random.randint(0, len(X_train)-1)
pattern = X_train[start]
# print ('pattern.shape', pattern.shape)
noise = np.random.normal(0, 1, (1, sequence_length))
predict_vel = generator2.predict(noise)
predict_vel = np.squeeze(predict_vel)
predict_vel = np.squeeze(scaler.inverse_transform(predict_vel.reshape(-1,1)))
predict_vel = [int(i) for i in predict_vel]
# print(predict_vel)
#############################################################################################



############################# music File saving ###########################################
midif = MidiFile()
track = MidiTrack()
t = 0
for n,v in zip(predict_notes,predict_vel):
    note=np.asarray([147,abs(n),abs(v)])
    bytes = note.astype(int)
    print(bytes)
    msg = Message.from_bytes(bytes[0:3])
    t += 1
    msg.time = t
    track.append(msg)
midif.tracks.append(track)
midif.save('LSTM_music.mid')

