# Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling

This is a modified repository for testing purposes.

The original Repository can be found at:
https://github.com/NJUNLP/TOWE

You can find the paper at:
https://www.aclweb.org/anthology/N19-1259

## Dependency:
* python3
* pytorch

## How to run (Take the 14res dataset as example)

1. (optional) prepare the word embeddings: (we have prepared for you in the code/data/ directory.)
    1. put the glove embeddings in the code/embedding directory.
    2. run the script:
    ```
    python build_vocab_embed.py --ds 14res
    ```
2. To train the model write:
    ```
    python main.py --ds 14res
    ```
3. To test the model:
    ```
    python main.py --ds 14res --test 1
    ```
    Put the sentence you want to test in "test_file.txt".
    Put the target words in the file "target_words.txt".
    The results will appear in 'results.txt' file.

    You might have to change the name of the model to your trained model. For
    that open main.py. Go to test(). Change the model name there.

    This is not perfect. But it suffices the purpose and it works :P :D

    There are a lot of options here. You can explore the options in main.py
    All the arguments have been listed there.
