"""
This module provides a simple dataloader for downloading
and preparing the TinyShakespeare dataset for training.
"""

import os
import requests

import torch
import numpy as np

def download_data(save_path: str = 'data.txt') -> list:
  """
  Downloads the TinyShakespeare dataset.

  Args:
    save_path (str): path to where the data will be downloaded

  Returns:
    torch.Tensor : TinyShakespeare dataset
  """
  url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
  content = requests.get(url).content
  with open(save_path, 'wb') as f:
    f.write(content)
  with open(save_path, 'r', encoding='utf-8') as f:
    dataset = f.read()
  return dataset

class Tokenizer:
  """
  Tokenizer based on an input character list

  This class provides functionality for converting text
  to character-level tokens.
  """
  def __init__(self, unique_characters: str):
    self.vocab = unique_characters
    self.character_to_index = {ch:i for i,ch in enumerate(self.vocab)}
    self.index_to_character = {i:ch for i,ch in enumerate(self.vocab)}

  def encode(self, text: str) -> list[int]:
    """
    Encode the input text into token indices

    Args:
      text (str): string to tokenize

    Returns:
      list: list of corresponding token indices
    """
    return [self.character_to_index[c] for c in text]

  def decode(self, indices: list[int]) -> str:
    """
    Decode the list of token indices into a string

    Args:
      indices (list[int]): list of integer token indices

    Returns:
      str: corresponding string
    """
    return "".join([self.index_to_character[i] for i in indices])

  def get_vocab_size(self):
    """Returns the size of the vocabulary"""
    return len(self.vocab)

class NTPDataset(torch.utils.data.Dataset):
  """
  Next Token Prediction Dataset
  """

  def __init__(self, data:torch.Tensor, config):
    super().__init__()
    self.data = data
    self.context_length = config.block_size
    self.device = config.device

  def __len__(self):
    return self.data.shape[0] - self.context_length

  def __getitem__(self, index):
    """
    Returns the context following the index, and the 
    associated next token prediction targets.
    """
    context = self.data[index:index+self.context_length]
    targets = self.data[index+1:index+1+self.context_length]
    return context.to(self.device), targets.to(self.device)
  
def prepare_data(config) -> tuple[NTPDataset, NTPDataset, Tokenizer]:
  """
  Prepares dataset for model training

  Converts the original dataset in the form of a string
  into two sequences of token indices -- one for training
  and one for testing.

  Args:
    config (GPTConfig): config specifying dataset parameters

  Returns:
    tuple(
      train_dataset (BaseDataset):
      valid_dataset (BaseDataset):
    )
    tokenizer (Tokenizer)
  """

  # download data and construct tokenizer
  dataset = download_data()
  vocab = sorted(list(set(dataset)))
  tokenizer = Tokenizer(vocab)

  # tokenize dataset (token indices)
  data = torch.tensor(
      tokenizer.encode(dataset),
      dtype=torch.long,
  )
  
  # training fraction (currently hardcoded)
  num_train = int(config.train_frac * len(data))

  train_dataset = NTPDataset(
    data[:num_train],
    config,
  ) 

  valid_dataset = NTPDataset(
    data[num_train:],
    config,
  ) 

  return train_dataset, valid_dataset, tokenizer
            