# CodingThreeFinal

Yan Wang 22019755

code：https://github.com/tomoko-tiba/CodingThreeFinal/blob/main/Coding3Final_Generate_Song.ipynb

data：https://github.com/tomoko-tiba/CodingThreeFinal/blob/main/songs

## Project Overview

This project aims to train a lyrics generator using TensorFlow and LSTM models on the entire collection of lyrics by the popular singer Taylor Swift. By employing deep learning techniques, the generator will be able to learn Taylor Swift's unique lyric style and themes, and generate new lyrics that closely resemble her style. This project will serve as a creative tool for songwriting enthusiasts, while also showcasing the application of deep learning in the music domain.

## Project Workflow

### Preliminary Research: 

Conduct initial research on natural language processing (NLP) techniques and deep learning models for text generation. Explore related projects and resources, such as the article "NLP in TensorFlow: Generate an Ed Sheeran Song" available at https://towardsdatascience.com/nlp-in-tensorflow-generate-an-ed-sheeran-song-8f99fe76662d, which provides valuable insights into generating song lyrics using TensorFlow and NLP.

### Preliminary Debug：

After obtaining the reference code, I attempted to train it using the lyrics data provided in the blog. However, I encountered three bugs that prevented smooth execution.

***Bug1***
```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[8], line 7
      5 token_list = tokenizer.texts_to_sequences([seed_text])[0]
      6 token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
----> 7 predicted = model.predict_classes(token_list, verbose=0)
      8 output_word = ""
      9 #print(predicted)

AttributeError: 'Sequential' object has no attribute 'predict_classes'
```
In the latest version of Keras, the predict_classes method has been deprecated and removed.

The alternative approach is to use the predict method to obtain the output probability distribution of the model and select the most likely class in an appropriate manner.

```python
# After modification
predicted_probabilities = model.predict(token_list, verbose=0)
```
***Bug2***

```python
adam = Adam(lr=0.01)
# WARNING:absl:`lr` is deprecated in Keras optimizer, please use `learning_rate` or use the legacy optimizer, e.g.,tf.keras.optimizers.legacy.Adam.
```
This warning message occurs when using the Keras optimizer. It notifies me that the "lr" parameter is deprecated and no longer recommended to use. Instead, I should use the "learning_rate" parameter to specify the learning rate.

```python
# After modification
adam = Adam(learning_rate=0.01)
```

***Bug3***

```python
for i in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
  predicted = model.predict_classes(token_list, verbose=0)
  output_word = ""
  #print(predicted)
  for word,index in tokenizer.word_index.items():
    #print(word)
    if index == predicted:
      output_word = word
      break
  seed_text += " " + output_word
print(seed_text)
```
```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[9], line 12
      9 #print(predicted)
     10 for word,index in tokenizer.word_index.items():
     11   #print(word)
---> 12   if index == predicted:
     13     output_word = word
     14     break

ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```
This error occurs when you attempt to compare an array with a predicted value. In NumPy, using the == operator to compare arrays returns a boolean array indicating whether each element satisfies the condition. However, for a boolean array with multiple elements, Python cannot determine its truth value.

I have made the following modifications and optimizations to your code:

1. Changed the loop variable i to _ since it is not used within the loop body.
2. Used more descriptive variable names to improve code readability.
3. Used np.argmax to obtain the predicted class instead of the deprecated model.predict_classes method.
4. Saved the predicted probability distribution as predicted_probabilities instead of using just predicted as the variable name.
5. Moved the string concatenation operation seed_text += " " + output_word to the end of the loop body.
   
These modifications and optimizations enhance the code's readability, efficiency, and compatibility with the latest version of the Keras library.

```python
# After modification
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probabilities = model.predict(token_list, verbose=0)
    predicted_class = np.argmax(predicted_probabilities)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_class:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
```

### Data Collection and Preprocessing: 

Collect the entire collection of Taylor Swift's lyrics from the webpage "Taylor Swift - All Lyrics (30 Albums)" available at https://www.kaggle.com/datasets/ishikajohari/taylor-swift-all-lyrics-30-albums. This dataset provides a comprehensive collection of Taylor Swift's lyrics from 30 albums. 

Due to this dataset, where the lyrics of each song are individually stored in separate text files, I manually merged all the lyrics into one file. At the same time, I cleaned the data by removing irrelevant characters such as ‘ [Pre-Chorus], [Chorus], [Verse 2] ’, and repetitive interjections like ‘ (Oh, oh, oh, oh, oh, oh, oh, oh, oh, oh) ’. This ensures that the final generated output is not influenced by them.

### Training Attempt: 

After going through the aforementioned preliminary process, I finally began the first training session. However, due to the reference blog data containing only 2,102 lines of lyrics, while the final compiled dataset consisted of approximately 11,000 lines of lyrics, the training time significantly increased.

```python
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
history = model.fit(predictors, label, epochs=100, verbose=1)
#print model.summary()
print(model)

# ref：https://towardsdatascience.com/nlp-in-tensorflow-generate-an-ed-sheeran-song-8f99fe76662d
```

After a long wait, the computer completed 100 epochs, and the first training results emerged:

![结果1](https://github.com/tomoko-tiba/CodingThreeFinal/assets/41440180/1a608eb9-2e19-4b83-b00b-35f16383ed98)

However, the final results were highly unsatisfactory. From a quantitative perspective, the accuracy score was only around 0.45, whereas it reached approximately 0.9 when running the data from the blog. Additionally, the loss value was around 2.45, which is relatively high.

Examining the generated text, it was evident that the model frequently failed to form complete words. This severely impacted the coherency and overall quality of the generated lyrics.

```python
seed_text = "What love is"
next_words = 60
# Result：ohncesscommunityirtsat i love is a chance

seed_text = "HI"
next_words = 60
# Result：parallel wireheroolee who i could be the man who'd throw

seed_text = "Just close your eyes"
next_words = 100
# Result：anymoreviewsaysr kids mondaysardwalk but they follow follow you home
```

At first, I attempted to adjust the learning_rate by using lower values, but there was no significant improvement in the results. Later on, while reviewing the code repeatedly, I noticed that many words in my data contained the carriage return character '\r'. I suspected that this character was causing the generation of incorrect words.
```python
tokenizer = Tokenizer()
data = open(path_to_file, 'rb').read().decode(encoding='utf-8')
corpus = data.lower().split("\n")

#corpus[2090:2101]
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

#tokenizer.word_index
print(tokenizer.word_index)
print(total_words)
# {'you': 1, 'i': 2, 'the': 3, 'and': 4, '\r': 5, 'to': 6, 'a': 7, 'in': 8, 'it': 9, 'my': 10, 'me': 11, 'your': 12, 'of': 13, 'that': 14, "i'm": 15, 'but': 16, 'all': 17, 'like': 18, 'on': 19, 'we': 20, 'oh': 21, 'know': 22, 'is': 23, 'was': 24, 'be': 25, 'this': 26, "it's": 27, 'so': 28, 'just': 29, 'when': 30, 'for': 31, "don't": 32, 'never': 33, 'you\r': 34, 'what': 35, "you're": 36, 'with': 37, 'me\r': 38, 'at': 39, 'if': 40, 'up': 41, "'cause": 42, 'love': 43, 'no': 44, 'now': 45, 'back': 46, 'one': 47, 'they': 48, 'were': 49, 'might': 50, 'time': 51, 'out': 52, 'got': 53, 'think': 54, 'like\r': 55, 'ooh': 56, 'see': 57, 'could': 58, 'say': 59, 'are': 60, 'not': 61, 'been': 62, 'also': 63, 'can': 64, 'wanna': 65, 'he': 66, 'come': 67, "can't": 68, "i'll": 69, 'ever': 70, 'get': 71, 'do': 72, 'have': 73, 'how': 74, 'want': 75, 'had': 76, 'from': 77, 'there': 78, 'said': 79, 'take': 80, 'still': 81, 'go': 82, 'right': 83, 'about': 84, 'look': 85, 'she': 86, 'did': 87, "i'd": 88, 'down': 89, 'would': 90, "i've": 91, 'gonna': 92, 'yeah': 93, 'baby': 94, 'as': 95, 'oh\r': 96, "that's": 97, 'made': 98, 'tell': 99, 'time\r': 100, 'it\r': 101, "didn't": 102, 'stay': 103, 'too': 104, 'last': 105, 'down\r': 106, 'now\r': 107, 'here': 108... .... 
```

I tried removing the carriage return characters to see if it would optimize the generated results.
```python
# After modification
tokenizer = Tokenizer()
data = open(path_to_file, 'rb').read().decode(encoding='utf-8')
cleaned_data = data.replace("\r", "")
corpus = data.lower().split("\n")
```

The second iteration showed significant improvement, with the accuracy value increasing from 0.45 to 0.65, and the loss value decreasing from 2.4 to 1.4.
```
Epoch 1/50
2178/2178 [==============================] - 69s 31ms/step - loss: 5.2726 - accuracy: 0.1505
Epoch 2/50
2178/2178 [==============================] - 67s 31ms/step - loss: 3.7426 - accuracy: 0.3172
Epoch 3/50
2178/2178 [==============================] - 67s 31ms/step - loss: 2.9197 - accuracy: 0.4199
Epoch 4/50
2178/2178 [==============================] - 65s 30ms/step - loss: 2.4214 - accuracy: 0.4890
Epoch 5/50
2178/2178 [==============================] - 68s 31ms/step - loss: 2.1050 - accuracy: 0.5380
Epoch 6/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.9059 - accuracy: 0.5700
Epoch 7/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.7813 - accuracy: 0.5920
Epoch 8/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.6895 - accuracy: 0.6083
Epoch 9/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.6179 - accuracy: 0.6210
Epoch 10/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.5682 - accuracy: 0.6298
Epoch 11/50
2178/2178 [==============================] - 70s 32ms/step - loss: 1.5371 - accuracy: 0.6319
Epoch 12/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.5070 - accuracy: 0.6360
Epoch 13/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4893 - accuracy: 0.6416
Epoch 14/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4681 - accuracy: 0.6447
Epoch 15/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4738 - accuracy: 0.6434
Epoch 16/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4648 - accuracy: 0.6428
Epoch 17/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4382 - accuracy: 0.6508
Epoch 18/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4583 - accuracy: 0.6442
Epoch 19/50
2178/2178 [==============================] - 69s 32ms/step - loss: 1.4334 - accuracy: 0.6481
Epoch 20/50
2178/2178 [==============================] - 69s 32ms/step - loss: 1.4262 - accuracy: 0.6506
Epoch 21/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4470 - accuracy: 0.6446
Epoch 22/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4365 - accuracy: 0.6481
Epoch 23/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4325 - accuracy: 0.6469
Epoch 24/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4099 - accuracy: 0.6522
Epoch 25/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4190 - accuracy: 0.6508
Epoch 26/50
2178/2178 [==============================] - 69s 32ms/step - loss: 1.4120 - accuracy: 0.6521
Epoch 27/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4148 - accuracy: 0.6509
Epoch 28/50
2178/2178 [==============================] - 69s 32ms/step - loss: 1.4071 - accuracy: 0.6528
Epoch 29/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4082 - accuracy: 0.6532
Epoch 30/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4289 - accuracy: 0.6475
Epoch 31/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4023 - accuracy: 0.6540
Epoch 32/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4127 - accuracy: 0.6505
Epoch 33/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4091 - accuracy: 0.6503
Epoch 34/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4153 - accuracy: 0.6521
Epoch 35/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4589 - accuracy: 0.6410
Epoch 36/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4266 - accuracy: 0.6488
Epoch 37/50
2178/2178 [==============================] - 68s 31ms/step - loss: 1.4671 - accuracy: 0.6381
Epoch 38/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4340 - accuracy: 0.6435
Epoch 39/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4109 - accuracy: 0.6491
Epoch 40/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4300 - accuracy: 0.6463
Epoch 41/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4180 - accuracy: 0.6491
Epoch 42/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4053 - accuracy: 0.6518
Epoch 43/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4177 - accuracy: 0.6501
Epoch 44/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4238 - accuracy: 0.6463
Epoch 45/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4260 - accuracy: 0.6454
Epoch 46/50
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4000 - accuracy: 0.6526
Epoch 47/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4053 - accuracy: 0.6535
Epoch 48/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4211 - accuracy: 0.6474
Epoch 49/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4209 - accuracy: 0.6475
Epoch 50/50
2178/2178 [==============================] - 67s 31ms/step - loss: 1.4147 - accuracy: 0.6512
```
![download](https://github.com/tomoko-tiba/CodingThreeFinal/assets/41440180/27464e6d-9677-49dc-81f9-15dfb6771339)
![1](https://github.com/tomoko-tiba/CodingThreeFinal/assets/41440180/934d3b02-442a-4519-b560-287dfbaf36a5)


### Result 1:
```
Just close your eyes
looking into me like me
makes me a note on the restaurant in my hair
when the sleep for the face would hang like
i thought i say i did something to her but all went down down
just wrong now was nothing
just the good oh yeah sounds days
but i'm not down in the dark
oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh
oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh oh
```

```
What is love it all good love
permanent damage you decide to me around this way
i feel like weapons
until i woke like the water
there's graded on the time not without me to me
now the house was the weather
got the time taken up the sky
my friends talk to the summit of a dreamer me my places on your
```

```
I hate you turn right around
and turn my daughter make it could be love
love at your first dancing when the one too the fight
i was get the tracks your head on my sleeve
and then they get the power clap for the ages
who was a man girl
hits made us now got 'em without time
sky your face when
```

```
I don't know what I want
you bless my name on me in a dream
or my love yeah yeah oh yeah yeah
ooh the aholic
but they want from me like time
it's just pretend it isn't it isn't
what you want call it
what you want to drive up
to your face
in the mirror like the others
start home turned down you pass
```

### Evolution

在接下来中，我设置了更低的learning_rate为0.001，并将epochs设置为100。
```
Epoch 1/100
2178/2178 [==============================] - 67s 30ms/step - loss: 5.7996 - accuracy: 0.0807
Epoch 2/100
2178/2178 [==============================] - 66s 30ms/step - loss: 4.7744 - accuracy: 0.1652
Epoch 3/100
2178/2178 [==============================] - 66s 30ms/step - loss: 4.1149 - accuracy: 0.2411
Epoch 4/100
2178/2178 [==============================] - 66s 30ms/step - loss: 3.5862 - accuracy: 0.3135
Epoch 5/100
2178/2178 [==============================] - 66s 30ms/step - loss: 3.1554 - accuracy: 0.3781
Epoch 6/100
2178/2178 [==============================] - 69s 31ms/step - loss: 2.7986 - accuracy: 0.4354
Epoch 7/100
2178/2178 [==============================] - 68s 31ms/step - loss: 2.5049 - accuracy: 0.4844
Epoch 8/100
2178/2178 [==============================] - 66s 30ms/step - loss: 2.2569 - accuracy: 0.5299
Epoch 9/100
2178/2178 [==============================] - 66s 30ms/step - loss: 2.0486 - accuracy: 0.5673
Epoch 10/100
2178/2178 [==============================] - 65s 30ms/step - loss: 1.8731 - accuracy: 0.5990
Epoch 11/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.7231 - accuracy: 0.6288
Epoch 12/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.5970 - accuracy: 0.6520
Epoch 13/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.4863 - accuracy: 0.6732
Epoch 14/100
2178/2178 [==============================] - 65s 30ms/step - loss: 1.3915 - accuracy: 0.6898
Epoch 15/100
2178/2178 [==============================] - 65s 30ms/step - loss: 1.3108 - accuracy: 0.7057
Epoch 16/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.2385 - accuracy: 0.7190
Epoch 17/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.1760 - accuracy: 0.7322
Epoch 18/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.1203 - accuracy: 0.7427
Epoch 19/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.0753 - accuracy: 0.7504
Epoch 20/100
2178/2178 [==============================] - 66s 30ms/step - loss: 1.0308 - accuracy: 0.7608
Epoch 21/100
2178/2178 [==============================] - 66s 30ms/step - loss: 0.9969 - accuracy: 0.7659
Epoch 22/100
2178/2178 [==============================] - 68s 31ms/step - loss: 0.9627 - accuracy: 0.7733
Epoch 23/100
2178/2178 [==============================] - 68s 31ms/step - loss: 0.9342 - accuracy: 0.7766
Epoch 24/100
2178/2178 [==============================] - 69s 32ms/step - loss: 0.9103 - accuracy: 0.7813
Epoch 25/100
2178/2178 [==============================] - 70s 32ms/step - loss: 0.8903 - accuracy: 0.7851
Epoch 26/100
2178/2178 [==============================] - 70s 32ms/step - loss: 0.8703 - accuracy: 0.7876
Epoch 27/100
2178/2178 [==============================] - 65s 30ms/step - loss: 0.8530 - accuracy: 0.7910
Epoch 28/100
2178/2178 [==============================] - 66s 30ms/step - loss: 0.8378 - accuracy: 0.7934
Epoch 29/100
2178/2178 [==============================] - 66s 30ms/step - loss: 0.8264 - accuracy: 0.7946
Epoch 30/100
2178/2178 [==============================] - 66s 30ms/step - loss: 0.8128 - accuracy: 0.7970
Epoch 31/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.8047 - accuracy: 0.7987
Epoch 32/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7942 - accuracy: 0.8002
Epoch 33/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7848 - accuracy: 0.8015
Epoch 34/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7783 - accuracy: 0.8016
Epoch 35/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7716 - accuracy: 0.8019
Epoch 36/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7643 - accuracy: 0.8037
Epoch 37/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7599 - accuracy: 0.8040
Epoch 38/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7535 - accuracy: 0.8051
Epoch 39/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7503 - accuracy: 0.8045
Epoch 40/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7469 - accuracy: 0.8044
Epoch 41/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7423 - accuracy: 0.8049
Epoch 42/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7373 - accuracy: 0.8061
Epoch 43/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7342 - accuracy: 0.8063
Epoch 44/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7329 - accuracy: 0.8054
Epoch 45/100
2178/2178 [==============================] - 70s 32ms/step - loss: 0.7312 - accuracy: 0.8051
Epoch 46/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7272 - accuracy: 0.8053
Epoch 47/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7246 - accuracy: 0.8067
Epoch 48/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7246 - accuracy: 0.8062
Epoch 49/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7209 - accuracy: 0.8064
Epoch 50/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7188 - accuracy: 0.8067
Epoch 51/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7178 - accuracy: 0.8060
Epoch 52/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7153 - accuracy: 0.8071
Epoch 53/100
2178/2178 [==============================] - 65s 30ms/step - loss: 0.7136 - accuracy: 0.8072
Epoch 54/100
2178/2178 [==============================] - 64s 30ms/step - loss: 0.7121 - accuracy: 0.8077
Epoch 55/100
2178/2178 [==============================] - 64s 30ms/step - loss: 0.7111 - accuracy: 0.8075
Epoch 56/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7106 - accuracy: 0.8069
Epoch 57/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7068 - accuracy: 0.8070
Epoch 58/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7065 - accuracy: 0.8068
Epoch 59/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7065 - accuracy: 0.8064
Epoch 60/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7054 - accuracy: 0.8077
Epoch 61/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7040 - accuracy: 0.8071
Epoch 62/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.7028 - accuracy: 0.8067
Epoch 63/100
2178/2178 [==============================] - 64s 29ms/step - loss: 0.7020 - accuracy: 0.8073
Epoch 64/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6996 - accuracy: 0.8076
Epoch 65/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.7016 - accuracy: 0.8077
Epoch 66/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6989 - accuracy: 0.8078
Epoch 67/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6973 - accuracy: 0.8077
Epoch 68/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6993 - accuracy: 0.8065
Epoch 69/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6962 - accuracy: 0.8077
Epoch 70/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6982 - accuracy: 0.8077
Epoch 71/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6953 - accuracy: 0.8080
Epoch 72/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6962 - accuracy: 0.8070
Epoch 73/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6959 - accuracy: 0.8074
Epoch 74/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6940 - accuracy: 0.8080
Epoch 75/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6933 - accuracy: 0.8082
Epoch 76/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6937 - accuracy: 0.8079
Epoch 77/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6923 - accuracy: 0.8082
Epoch 78/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6918 - accuracy: 0.8083
Epoch 79/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6920 - accuracy: 0.8082
Epoch 80/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6934 - accuracy: 0.8075
Epoch 81/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6911 - accuracy: 0.8079
Epoch 82/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6916 - accuracy: 0.8069
Epoch 83/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6893 - accuracy: 0.8079
Epoch 84/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6899 - accuracy: 0.8077
Epoch 85/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6884 - accuracy: 0.8079
Epoch 86/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6878 - accuracy: 0.8075
Epoch 87/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6899 - accuracy: 0.8077
Epoch 88/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6876 - accuracy: 0.8076
Epoch 89/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6874 - accuracy: 0.8079
Epoch 90/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6886 - accuracy: 0.8079
Epoch 91/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6893 - accuracy: 0.8087
Epoch 92/100
2178/2178 [==============================] - 63s 29ms/step - loss: 0.6845 - accuracy: 0.8089
Epoch 93/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6862 - accuracy: 0.8078
Epoch 94/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6861 - accuracy: 0.8084
Epoch 95/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6858 - accuracy: 0.8080
Epoch 96/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6856 - accuracy: 0.8084
Epoch 97/100
2178/2178 [==============================] - 62s 29ms/step - loss: 0.6855 - accuracy: 0.8088
Epoch 98/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6846 - accuracy: 0.8079
Epoch 99/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6878 - accuracy: 0.8077
Epoch 100/100
2178/2178 [==============================] - 62s 28ms/step - loss: 0.6854 - accuracy: 0.8083
<keras.engine.sequential.Sequential object at 0x000001D6B0B379D0>
```
![1](https://github.com/tomoko-tiba/CodingThreeFinal/assets/41440180/cfacb959-5f27-4076-9256-a08422387703)
![2](https://github.com/tomoko-tiba/CodingThreeFinal/assets/41440180/b2e28844-375a-4b0f-87c4-c67d576cabeb)

After adjusting the values, there was a noticeable improvement in the metrics. The accuracy reached 0.8, and the loss decreased to 0.68. The generated sentences also became more coherent and fluent in their structure.

### Result 2:
```
I don't know what I want
 so don't ask me at me that
hello to me like a back oh no
to mended to the man talk
they say i say to you losers
key in a but all you it was
just like me whole back
knew this won't gave there out of style
i'd say to run out things
we got a night in
```

```
What is love me changed like that 
a green light  
taken your name on me 
your past me me me 
your back now it could be the one that all of you i know 
and it could be the mean that you're my door 
but if you're bluff sized around you 
my friends talk to my back to 
to me straight to me
```

```
Just close your eyes 
like we're rode to do it in time that way
it now it would be the things that it was stealing baby 
oh yeah oh yeah you say yeah yeah yeah yeah yeah yeah we
and come oh oh and to a white horse 
yes sellin' swift yeah ayy hoo ayy yeugh 
chances to your face friends make plans of
```

## Conclusion:
By implementing the LSTM model and training it on Taylor Swift's extensive collection of lyrics, the lyrics generator has successfully learned the stylistic elements and themes characteristic of Taylor Swift's songs. The generator can now produce lyrics that closely resemble her unique style. Although some generated sentences may lack fluency, the overall readability has been noticeably enhanced.

Moving forward, my focus will be on refining the model and exploring additional techniques to address the remaining challenges. I will continue to delve into advanced optimization methods to further enhance the coherence and quality of the generated lyrics. By studying and implementing state-of-the-art approaches, I aim to push the boundaries of what the lyrics generator can achieve.

In conclusion, while there is still room for improvement in terms of sentence coherence, the project has demonstrated great progress in enhancing the readability of the generated text. With a commitment to ongoing learning and optimization, I am determined to overcome the current limitations and create a more refined and impressive lyrics generator in the future.
