<div align="center">
  <a href="https://github.com/tensorflow/tensorflow">
    <img src="https://img.shields.io/badge/TensorFlow-FF8000?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  </a>
  <a href="https://keras.io/">
    <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" alt"Keras">
  </a>
  <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">
    <img src="https://img.shields.io/badge/LSTM-009900?style=for-the-badge" alt"LSTM">
  </a>
  <a href="https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average">
    <img src="https://img.shields.io/badge/ARIMA-0080FF?style=for-the-badge" alt"ARIMA">
  </a>
</div>

<hr/>

# Neural Stock Prophet

## Abstract
Stock Price Prediction using a machine learning algorithm helps discover the future value of company stock and other financial assets traded on an exchange. Whereas, the existing methods relied highly on model setup and tuning, without considering the variation of data. Also, the machine learning model faces the problems of overfitting and performance limitations. The proposed solution combines several techniques and algorithms to enhance and evaluate the robustness, stability, and interoperability of the stock price prediction algorithm.
* LSTM model with attention mechanisms
* Multiplicative Decomposition
* ARIMA model
* Risk Parity Portfolio

<hr/>

## Dependencies
You can install all the packages via
```
pip install -r requirements.txt
```

<hr/>

## Instructions
There are lots of parameters required to construct several algorithms. For more details on the parameters, please visit [Parameters](#parameters)

For simple usage, you can just run
```
python3 main.py
```

It is suggested to change different parameters to know the effects  
  
E.g. Timesteps = 30 for dataset construction, window lengths = 252 for multiplicative decomposition, arima order = (2, 0, 2) for ARIMA model construction, and use the historical stock price from 1st January 2012 to 31st December 2022 to train the model, and use the stock price from 1st January 2023 to 31st December 2023 to perform forecasting. 
```
python3 main.py --time_steps=30 --window_lengths=252 --arima_order 2 0 2 --train_date 2012-01-01 2022-12-31 --test_date 2023-01-01 2023-12-31
```

<hr/>

## Parameters
There are various parameters required by the algorithms. 
For more details, you can visit src/```args.py```  
The (.) after the argument type indicates the ```nargs=``` value in ```argparse.ArgumentParser().add_argument()``` method
| Parameter Name  | Meaning| Default Value| Type | Choice/Range |
| ---------- | -----------|-----------|-----------|-----------|
| stock_names   | List of stock names to forecast and analyze   |["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]   |str (+)   | (Valid stock name) |
|  train_date  | Start and end date for training data (in the format of YYYY-MM-DD)   | ["2016-01-01", "2022-12-31"]  |str (2)   | (Valid stock price date) | 
| test_date   | Start and end date for testing data (in the format of YYYY-MM-DD)   |["2023-01-01", "2023-12-31"]   |str (2)   | (Valid stock price date) | 
| keep_ratio   | Ratio of the number of features to keep during feature selection   |0.8   |float   | [0, 1] | 
| time_steps   | Number of time steps to consider for dataset construction   |60   |int   |[1, len(dataset)] | 
| window_lengths   | Length of the window for the seasonal component in multiplicative decomposition   |48   |int   | [1, len(dataset)] | 
| batch_size   | number of data per batch   |50   |int   | [1, total data size] | 
| epoch   | local epoch for client training   |5   |int   | [1, inf) | 
| weight_decay   | weight decay  |0.01   |float   | (0, inf) | 
| max_norm   | max norm for gradient clipping   |10.0   |float  | (0, inf) | 
| model_name   | model for training. The name is also the corresponding dataset name   |cifar10   |str   | {linear, mnist, emnist, cifar10, cifar100, resnet18, shakespeare} |
| rule   | the rule of data partitioning   |iid   |str   | {iid, dirichlet} |
|  rand_seed  | random seed   |1   |int   | [0, inf) | 
| save_period   | period to save the models   |1   |int   | [1, comm_rounds] |
| print_per   | period to print the training result   |5   |int   | [1, epoch] | 
| n_RIS_ele   | number of RIS elements   |40   |int   | [0, inf) |
| n_receive_ant   | number of receive antennas   |5   |int   | [0, inf) | 
| alpha_direct   | path loss component   |3.76   |float   | [0, inf) | 
| SNR   | noise variance/0.1W in dB   |90.0   |float   | [0, inf) | 
| location_range   | location range between clients and RIS   |30   |int   | [0, inf) | 
| Jmax   | number of maximum Gibbs Outer loops   |50   |int   | [1, inf) |
| tau   | the SCA regularization term   |0.03   |float   | [0, inf) | 
| nit   | I_max, number of maximum SCA loops   |100   |int   | [1, inf) | 
| threshold   | epsilon, SCA early stopping criteria   |0.01   |float   | [0, inf) | 
| transmit_power   | transmit power of clients   |0.003   |float   | [0, inf) | 
| noiseless   | whether the channel is noiseless   |False   |bool   | {True, False} | 
| rison   | whether the RIS is presented   |1   |int   | {0, 1} | 

<hr/>

## License

