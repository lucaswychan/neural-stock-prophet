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
+ LSTM model with attention mechanisms
+ Multiplicative Decomposition
+ ARIMA model
+ Risk Parity Portfolio

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
| factor   | Factor to combine the LSTM and ARIMA forecast, which control the importance of LSTM signal   |0.9   |float   | [0, 1] | 
| epochs   | Number of epochs for training the LSTM model   |50   |int   | [1, inf) | 
| batch_size   | Batch size for training the LSTM model  |32   |int   | [1, len(dataset)] | 
| lr   | Learning rate for training the LSTM model   |0.001   |float  | (0, inf) | 
| arima_order   | Order of the ARIMA model (requires 3 integers)   |[1, 0, 6]   |int (3)   | [0, inf) for 3 integers |
| arima_trend   | The deterministic trend in the ARIMA model   |ct   |str   | {n, c, t, ct} |
|  verbose  | Determine if saving and printing the result   |True   |bool   | {True, False} | 
| risk_distribution   | Determine the budget allocation of the portfolio   |eq   |str   | {eq, mv} |

<hr/>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<hr/>

## TODO
There are further improvements that can be made. Please have a look at the [TODO](TODO.md). 

<hr/>
