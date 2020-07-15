![nnDPI](https://drive.google.com/uc?export=view&id=1MAjeZqPWm3mAmPr3jULvQP2cwlVaBolq)

## What is nnDPI?
High performance deep packet inspection AI model using neural networks with an embedding layer, 1D Convolution layers and bidirectional gated recurrent unit.

### How was it trained?
- nnDPI was trained on VPN-nonVPN dataset (ISCXVPN2016) which consists of labeled network traffic, including full packet in pcap or pcapng format.
- As the dataset is unbalanced, the classes have been weighted and all samples have been used - no downsampling.​
- The model has been trained using Keras Tensorflow 
- Model has 796,418 trainable parameters only.​
- ADAM was used as an optimizer, with starting learning rate of 0.001​
- Learning rate is reduced over training epochs when no more improvements is being achieved.
- Batch size was set to 3072, which what could be fitted into 2 RTX 2080 GPUs memory, every GPU handles 1536 samples per batch, and then both GPUs add their results together.​
- 70 epochs (Early stopping)

### Dataset (ISCXVPN2016)
- Dataset available at: https://www.unb.ca/cic/datasets/vpn.html
- Uncompressed pcaps size: ~26GB
- After filtering irrelevant packets, we had ~19.5 million packets for training
- Stratified Train/Test/Validation splits were taken, with 80%, 10%, 10% respectively.

### What is the dataset packets content?
- Web Browsing: Firefox and Chrome
- Email: SMPTS, POP3S and IMAPS
- Chat: ICQ, AIM, Skype, Facebook and Hangouts
- Streaming: Vimeo and Youtube
- File Transfer: Skype, FTPS and SFTP using Filezilla and an external service
- VoIP: Facebook, Skype and Hangouts voice calls (1h duration)
- P2P: uTorrent and Transmission (Bittorrent)

### Which architecture was used?
nnDPI uses a mix of neural network layers, including:
- Word Embedding
- 1D Convolutions
- Batch Normalization
- Max Pooling
- RNN (Bidirectional GRU)
- Dense

### How to preprocess packets?
```
python nndpi_preprocessing.py --n_jobs=1 --pcap_dir=./CompletePCAPs/ --processed_pcap_dir=./ProcessedPackets/ --max_len=1500 --one_df=False
```

- set `pcap_dir` to the location of original pcap files, leave default if you would put the files in the CompletePCAPs dir.
- set `processed_pcap_dir` to the location to save the newely preprocessed dataframes, leave default to keep files in the ProcessedPackets dir
- set `n_jobs` to the number of workers to parallelize the process. (-1 to run workers on all cores) or to 1 if you want to run as a sequential process.
- set `max_len` to the preprocessed packet length in bytes
- set `one_df` to True if you want to save all the processed packets into a single dataframe. (Be careful, expensive RAM usage)

We have included a small sample of the dataset in this repo,  to get the full dataset please refer to the link above and download it.
We have also included the processed packets for this sample under ProcessedPackets dir for review.

### How to train the model?
```
python nndpi_train.py --multi_gpu=True --batch_size=3072 --max_len=1500
```
- set `multi_gpu` to True if you want to parallelize training on a multi gpu system.
- set `batch_size` to the training batch_size
- set `max_len` to the packet length specified during preprocessing

Training takes time and resources. This model was trained on a multi-gpu system with 64GB of RAM so there was no need to use generators as all data could fit into the memory.
If you have limited RAM, you might need to either use a generator or take a sample of the data.

### How to use the model?
The model is available under `./Model` dir in '.h5' format, you can load it with TF Keras.

With older TF versions, you might have problems loading the model directly as it was created with a multi gpu strategy, if so, please create the model architecture first with the `create_model` function then use `model.load_weights`

### DISCLAIMER
While we do our best to detect network traffic types from a single captured packed, we cannot guarantee that our software is error free and 100% accurate in traffic detection. 
Please respect the privacy of users and make sure you have proper authorization to listen, capture and inspect network traffic.
