# CodingThreeFinal

Yan Wang 22019755

## Project Overview

This project aims to train a lyrics generator using TensorFlow and LSTM models on the entire collection of lyrics by the popular singer Taylor Swift. By employing deep learning techniques, the generator will be able to learn Taylor Swift's unique lyric style and themes, and generate new lyrics that closely resemble her style. This project will serve as a creative tool for songwriting enthusiasts, while also showcasing the application of deep learning in the music domain.

## Project Workflow

### Preliminary Research: 

Conduct initial research on natural language processing (NLP) techniques and deep learning models for text generation. Explore related projects and resources, such as the article "NLP in TensorFlow: Generate an Ed Sheeran Song" available at https://towardsdatascience.com/nlp-in-tensorflow-generate-an-ed-sheeran-song-8f99fe76662d, which provides valuable insights into generating song lyrics using TensorFlow and NLP.

### Preliminary Debug：

After obtaining the reference code, I attempted to train it using the lyrics data provided in the blog. However, I encountered two bugs that prevented smooth execution.

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

"Due to this dataset, where the lyrics of each song are individually stored in separate text files, I manually merged all the lyrics into one file. At the same time, I cleaned the data by removing irrelevant characters such as ‘ [Pre-Chorus], [Chorus], [Verse 2] ’, and repetitive interjections like ‘ (Oh, oh, oh, oh, oh, oh, oh, oh, oh, oh) ’. This ensures that the final generated output is not influenced by them.

### First Training Attempt: 

在经过上述前期的过程之后


 
