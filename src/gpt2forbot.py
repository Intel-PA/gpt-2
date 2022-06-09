
#gpt2 init
from curses import raw
import sys
sys.path.append("/home/rehman/work/gpt-2/src")
models_dir = "/home/rehman/work/gpt-2/models"
current_model = "1558M"

import fire
import json
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import model, sample, encoder


class GPT2:

  
  # extracted from the source code to generate some text based on a prior
  def __init__(
      self,
      model_name='1558M',
      seed=None,
      nsamples=1,
      batch_size=1,
      length=20,
      temperature=1,
      top_k=40,
      raw_text="",
  ):
      """
      Interactively run the model
      :model_name=117M : String, which model to use
      :seed=None : Integer seed for random number generators, fix seed to reproduce
       results
      :nsamples=1 : Number of samples to return total
      :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
      :length=None : Number of tokens in generated text, if None (default), is
       determined by model hyperparameters
      :temperature=1 : Float value controlling randomness in boltzmann
       distribution. Lower temperature results in less random completions. As the
       temperature approaches zero, the model will become deterministic and
       repetitive. Higher temperature results in more random completions.
      :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
       considered for each step (token), resulting in deterministic completions,
       while 40 means 40 words are considered at each step. 0 (default) is a
       special setting meaning no restrictions. 40 generally is a good value.
      """
      if batch_size is None:
          batch_size = 1
      assert nsamples % batch_size == 0

      self.nsamples = nsamples
      self.batch_size = batch_size
      
      self.enc = encoder.get_encoder(model_name, models_dir)
      hparams = model.default_hparams()
      with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
          hparams.override_from_dict(json.load(f))

      if length is None:
          length = hparams.n_ctx // 2
      elif length > hparams.n_ctx:
          raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

      self.sess = tf.Session(graph=tf.Graph())
      self.sess.__enter__()
      
      self.context = tf.placeholder(tf.int32, [batch_size, None])
      np.random.seed(seed)
      tf.set_random_seed(seed)
      self.output = sample.sample_sequence(
          hparams=hparams, length=length,
          context=self.context,
          batch_size=batch_size,
          temperature=temperature, top_k=top_k
      )

      saver = tf.train.Saver()
      self.ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
      saver.restore(self.sess, self.ckpt)

  def close(self):
    self.sess.close()
  
  def generate_conditional(self,raw_text, last_length):
      context_tokens = self.enc.encode(raw_text)
      generated = 0
      #self.length = last_length
      for _ in range(self.nsamples // self.batch_size):
          out = self.sess.run(self.output, feed_dict={
              self.context: [context_tokens for _ in range(self.batch_size)]
          })[:, len(context_tokens):]
          for i in range(self.batch_size):
              generated += 1
              text = self.enc.decode(out[i])
              return text
              #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
              #print(text)
      #print("=" * 80)







#gpt2 = GPT2(model_name=current_model)
# you must also call download_model.py (see earlier cell) with the correct parameter
# 1558M, best results takes a long time to load
# 1558M, 774M, 355M, 345M, 124M, and 117M