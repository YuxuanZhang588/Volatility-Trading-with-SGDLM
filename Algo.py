from AlgorithmImports import *
from datetime import timedelta, datetime
from QuantConnect.DataSource import *
import json
import pandas as pd
import numpy as np
from Auxiliary_functions import SV_forecast

class FormalAsparagusFrog(QCAlgorithm):

    def Initialize(self):
        ### Basic Trading Environment Settings
        self.SetStartDate(2022,1,1)
        self.SetEndDate(2022,11,1)
        self.SetCash(10000000)
        
        ### Define a set of dictionaries to store values for each individual stock
        self.trade_dict = {}
        self.contract_dict = {}
        self.call_dict = {}
        self.put_dict = {}
        self.size_dict = {}
        self.previous_Delta_dict = {}
        self.type_dict = {}
        self.F_matrix_dict = {}
        self.r_t_dict = {}
        self.post_train_dict = {}
        self.alpha_dict = {'QCOM':11.2102,'AMAT':12.3144,'TSM':10.9097,'CRM':17.7513,'META':37.1196,'AMZN':14.4983,'NVDA':45.5533,'TSLA':21.5031,'XOM':32.1498, 'LLY': 32.1205}
        self.beta_dict = {'QCOM':0.6058,'AMAT':0.6309,'TSM':0.4052,'CRM':0.9390,'META':1.1821,'AMZN':0.5024,'NVDA':1.3147,'TSLA':1.2817,'XOM':1.7605, 'LLY': 1.8267}
        ### Call
        self.atm_call_contract_dict = {}
        self.atm_call_strike_dict = {}
        self.atm_call_impliedVol_dict = {}
        ### Put
        self.atm_put_contract_dict = {}
        self.atm_put_strike_dict = {}
        self.atm_put_impliedVol_dict = {}

        ### 'TSLA','META','LLY'
        self.tickers = ['TSLA','META', 'LLY']
        self.stock_symbol = {}
        self.option_symbol = {}
        self.day_dict = {key: 0 for key in self.tickers}
        for ticker in self.tickers:
            self.AddEquity(ticker,Resolution.Daily)
            option = self.AddOption(ticker,Resolution.Daily)
            option.SetFilter(-30, 30, 0, 31)
            self.stock_symbol[ticker] = (self.AddEquity(ticker,Resolution.Daily).Symbol)
            self.option_symbol[ticker] = (self.AddOption(ticker,Resolution.Daily).Symbol)

        ### Environment for Data Cleaning
        self.set_warm_up(200)
        self.wti = self.AddCfd("WTICOUSD", Resolution.Daily).Symbol
        self.set_risk_free_interest_rate_model(InterestRateProvider())
        self.vix_symbol = self.add_data(CBOE, 'VIX', Resolution.DAILY).symbol
        self.symbol_data = {}

        ### Variables for SV_Forecast
        cov_p = 7 # Number of Features in the Covariance Matrix
        self.G = np.diag([1]*cov_p)
        self.m0 = np.array([0]*cov_p)   
        self.C0 = np.diag([10000]*cov_p)
        self.J = 10
        self.q = np.array([0.0012, 0.0158, 0.0559, 0.1205, 0.1884, 0.2271, 0.2067, 0.1306, 0.0478, 0.0061])
        self.q /= np.sum(self.q)
        self.b = np.array([-7.325, -4.3419, -2.7762, -1.7339, -0.9864, -0.4259, 0.0113, 0.3675, 0.6737, 0.9634])
        self.w = np.array([1.8334, 1.0415, 0.6362, 0.3937, 0.2465, 0.1567, 0.1915, 0.0669, 0.0445, 0.0282])

        ### Risk Management
        self.port = self.Portfolio.TotalPortfolioValue
        self.max_drawdown_threshold = 0.1
        self.freeze = 0

        ### Transaction Cost Management
        self.contract_num_dict = {} # Track the Monthly Contract Number
        self.initial_cash = 10000000

    """
    This function Fetches the trained posterior data from Object Store
    """
    def FetchTrainedPost(self, ticker) -> None:
        ticker_low = ticker.lower()
        json_file = self.ObjectStore.GetFilePath('Team_1/Posteriors/Post_' + ticker_low + '.json')
        
        with open(json_file, 'r') as file:
            self.post_train_dict[ticker] = json.load(file)
            print(f"post train for tesla{self.post_train_dict[ticker]}")

    """
    This function Checks for Option Assignments
    (If Short Option and Buyer Exercise Right, Liquidate all positions)
    """
    def OnOrderEvent(self, orderEvent: OrderEvent) -> None:
        if orderEvent.Status == OrderStatus.Filled:
            for ticker, stock_symbol in self.stock_symbol.items():
                if orderEvent.Symbol == stock_symbol:
                    if self.trade_dict.get(ticker) == True: # Check if the trade flag is set for this stock
                        if orderEvent.is_assignment == True: # If Option is exercised, Liquidate all positions, and open for more trades
                            self.Log(f"Option assigned for {ticker} - liquidating stock position.")
                            self.Liquidate(stock_symbol) # Liquidate all Hedging Positions for this stock
                            self.Schedule.On(self.DateRules.Tomorrow, self.TimeRules.AfterMarketOpen(stock_symbol,0),
                                            lambda: self.Liquidate(stock_symbol))
                            self.trade_dict[ticker] = True
    
    """
    This function fetches the latest Option Chain with the farthest expiry
    """
    def FetchChain(self, slice: Slice, ticker, expiry=None):
        if(self.option_symbol.__contains__(ticker)):
            symbol = self.option_symbol[ticker]
            chain = slice.OptionChains.get(symbol)
            if chain:
                if expiry == None:
                    far_expiry = max([x.Expiry for x in chain]) # Get the Farthest Expiry within the Chain
                    self.timelag = (far_expiry - self.Time).days
                    expiry = far_expiry
                    self.Log(f'Checkpoint Expiry: {expiry}')
                call_contract_list = [x for x in chain if x.Expiry == expiry and x.Right==OptionRight.Call and x.Strike>=x.UnderlyingLastPrice]
                put_contract_list = [x for x in chain if x.Expiry == expiry and x.Right==OptionRight.Put and x.Strike<=x.UnderlyingLastPrice]
                if call_contract_list:
                    self.atm_call_contract_dict[ticker] = sorted(call_contract_list, key= lambda x: x.Strike)[0]
                    self.atm_call_strike_dict[ticker] = self.atm_call_contract_dict[ticker].Strike
                    self.atm_call_impliedVol_dict[ticker] = self.atm_call_contract_dict[ticker].ImpliedVolatility
                else:
                    self.atm_call_contract_dict[ticker] = self.atm_call_strike_dict[ticker] =  None
                    self.atm_call_impliedVol_dict[ticker] = 0.0
                    self.Log("No Call Contract List")
                if put_contract_list:
                    self.atm_put_contract_dict[ticker] = sorted(put_contract_list, key=lambda x: x.Strike)[0]
                    self.atm_put_strike_dict[ticker] = self.atm_put_contract_dict[ticker].Strike
                    self.atm_put_impliedVol_dict[ticker] = self.atm_put_contract_dict[ticker].ImpliedVolatility
                else:
                    self.atm_put_contract_dict[ticker] = self.atm_put_strike_dict[ticker] =  None
                    self.atm_put_impliedVol_dict[ticker] = 0.0
                    self.Log("No Put Contract List")
                return True
            else:
                self.Log("No Chain Fetched")
                return False

    """
    This function contains trading and dynamical hedging algorithms
    """
    def DoTrade(self, ticker, date, slice: Slice):
        currentStock = self.AddEquity(ticker, Resolution.Daily).Symbol

        if date == 0 :
            self.trade_dict[ticker] = True 
            self.contract_dict[ticker] = None 
            self.size_dict[ticker] = 30
            self.previous_Delta_dict[ticker] = None 
            self.type_dict[ticker] = None
            self.F_matrix_dict[ticker] = None
            self.r_t_dict[ticker] = None
            self.contract_num_dict[ticker] = 0

        if date % 30 == 0:
            self.Log(f'Num of Constract this month: {self.day_dict[ticker]}')
            self.day_dict[ticker] = 0

        if self.is_warming_up:
            self.F_matrix_dict[ticker], self.r_t_dict[ticker] = self.ProcessSymbolData(currentStock, self.F_matrix_dict[ticker], self.r_t_dict[ticker], slice)
        else:
            if self.trade_dict[ticker] and self.contract_num_dict[ticker] <= 50: # Open Initial Position, limit monthly transaction cost
                if self.FetchChain(slice, ticker) and self.timelag > 0:
                    self.FetchTrainedPost(ticker)
                    self.F_matrix_dict[ticker], self.r_t_dict[ticker] = self.ProcessSymbolData(currentStock, self.F_matrix_dict[ticker], self.r_t_dict[ticker], slice)
                    if self.F_matrix_dict[ticker] is None or self.r_t_dict[ticker] is None:
                        return
                    else:
                        F = self.F_matrix_dict[ticker][-(100 + self.timelag):]
                        R = self.r_t_dict[ticker][-100:]
                    ##### Use Model to Predict Volaliltiy at Expiry, expiry = self.far_expiry
                    print(f"post train for tesla{self.post_train_dict[ticker]}")
                    forecast = SV_forecast(R, F, self.G, self.m0, self.C0, delta=0.99, 
                                            J=self.J, q=self.q, b=self.b, w=self.w, u=1, 
                                            g=np.mean(self.post_train_dict[ticker]['mu_sample']), 
                                            GG=np.std(self.post_train_dict[ticker]['mu_sample'])**2, 
                                            c=np.mean(self.post_train_dict[ticker]['phi_sample']), 
                                            CC=np.std(self.post_train_dict[ticker]['phi_sample'])**2, 
                                            aa=2*self.alpha_dict[ticker], v0=self.beta_dict[ticker]/self.alpha_dict[ticker], n_burn_in=10, npost=100, K=self.timelag)
                    sigma_for = forecast['sigma_forecast'] # Forecast Realized Volatility
                    p90 = np.percentile(sigma_for, 90, axis=0)[-1:]
                    p10 = np.percentile(sigma_for, 10, axis=0)[-1:]
                    self.contract_dict[ticker] = None
                    
                    if self.atm_call_impliedVol_dict[ticker] > p90 and self.atm_put_impliedVol_dict[ticker] > p90 and self.atm_call_contract_dict[ticker] is not None and self.atm_put_contract_dict[ticker] is not None:
                        # High Vol, Long Straddle, Hedge the remaining delta with stock
                        self.call_dict[ticker] = self.atm_call_contract_dict[ticker]
                        self.put_dict[ticker] = self.atm_put_contract_dict[ticker]
                        self.contract_dict[ticker] = self.atm_call_contract_dict[ticker]
                        if self.securities[self.call_dict[ticker].symbol].is_tradable and self.securities[self.put_dict[ticker].symbol].is_tradable:
                            self.market_order(self.call_dict[ticker].Symbol, self.size_dict[ticker])
                            self.market_order(self.put_dict[ticker].Symbol, self.size_dict[ticker])
                            self.HedgeStraddle(self.call_dict[ticker], self.put_dict[ticker], ticker, currentStock, self.size_dict[ticker])

                    elif self.atm_put_impliedVol_dict[ticker] < p10 and self.atm_call_impliedVol_dict[ticker] < p10 and self.atm_put_contract_dict[ticker] is not None and self.atm_call_contract_dict[ticker] is not None:
                        # Low Vol, Long Put Long Stock
                        self.contract_dict[ticker] = self.atm_put_contract_dict[ticker]
                        if self.securities[self.contract_dict[ticker].symbol].is_tradable:
                            self.DeltaHedge(self.contract_dict[ticker], ticker, currentStock, self.size_dict[ticker])

            else: # There exist option positions in portfolio, start Dynamic Hedging
                self.FetchChain(slice, ticker, self.contract_dict[ticker].Expiry)
                if self.type_dict[ticker] == 0:
                    self.call_dict[ticker] = self.atm_call_contract_dict[ticker]
                    self.put_dict[ticker] = self.atm_put_contract_dict[ticker]
                    current_delta = self.call_dict[ticker].Greeks.Delta + self.put_dict[ticker].Greeks.Delta
                    if abs(int(current_delta-self.previous_Delta_dict[ticker])*self.size_dict[ticker]*100) > 0:
                        self.market_order(stock, -int(100*(current_delta-self.previous_Delta_dict[ticker])*size))
                        self.previous_Delta_dict[ticker] = current_delta

                else:
                    self.contract_dict[ticker] = self.atm_put_contract_dict[ticker]
                    self.Debug(f'PreviousDelta for {currentStock}: {self.previous_Delta_dict[ticker]}')
                    self.Debug(f'CurrentDelta for {currentStock}: {self.contract_dict[ticker].Greeks.Delta}')

                # Make sure the position difference is greater at least 1
                    if not int(100*self.size_dict[ticker]*(self.contract_dict[ticker].Greeks.Delta-self.previous_Delta_dict[ticker])) == 0:
                        self.MarketOrder(currentStock, int(100*self.size_dict[ticker]*(self.contract_dict[ticker].Greeks.Delta-self.previous_Delta_dict[ticker])))
                        self.previous_Delta_dict[ticker] = self.contract_dict[ticker].Greeks.Delta
                # On the day of Expiry, Liquidate all positions
                if self.Time == self.contract_dict[ticker].Expiry:
                    self.Log("OTM Exercise, Time to Liquidate")
                    self.Liquidate(currentStock)
                    self.trade_dict[ticker] = True
    
    """
    This function makes contract order and stock order with the corresponding delta hedged position
    """
    def DeltaHedge(self,contract,ticker,stock, size):
        self.MarketOrder(contract.Symbol, size)
        self.Log(f'Purchased Option Expiry: {self.contract_dict[ticker].Expiry}')
        delta = contract.Greeks.Delta
        self.MarketOrder(stock, int(100*delta*size))
        self.trade_dict[ticker] = False
        self.previous_Delta_dict[ticker] = delta
        self.type_dict[ticker] = 1

    """
    This function hedges the remaining delta position of long straddle with stock
    """
    def HedgeStraddle(self, call_contract, put_contract, ticker, stock, size):
        delta = call_contract.Greeks.Delta + put_contract.Greeks.Delta
        if abs(-int(100*delta*size)) != 0:
            self.MarketOrder(stock, -int(100*delta*size))
            self.trade_dict[ticker] = False
            self.previous_Delta_dict[ticker] = delta
        self.type_dict[ticker] = 0

    """
    This is the main OnData function, If all risk and transaction conditions are checked, start trade
    """
    def OnData(self, slice: Slice) -> None:
        if self.freeze != 0:
            self.freeze -= 1
            self.debug(f'Freezing, {self.freeze}')
            return
        current_value = self.Portfolio.TotalPortfolioValue
        if current_value > self.port:
            self.port = current_value
        else:
            current_drawdown = (self.port - current_value) / self.port
            if current_drawdown > self.max_drawdown_threshold:
                self.Liquidate()
                self.freeze = 30
                self.port = current_value
                self.Debug(f"Liquidating due to drawdown of {current_drawdown*100:.2f}% exceeding threshold.")
                for ticker in self.tickers:
                    self.trade_dict[ticker] = True
                return

        if self.Time.hour == 0 and self.Time.minute == 0 and self.Time.second == 0:
            for ticker in self.tickers:
                self.DoTrade(ticker, self.day_dict[ticker], slice)
                self.day_dict[ticker] += 1

        if self.Portfolio.Invested and (self.initial_cash - self.port) > self.initial_cash * 0.2:
                self.Liquidate()
                self.Log("Stop loss triggered. Liquidating positions.")
                self.initial_cash = self.portfolio.cash  # Reset initial investment
                self.trade = True

    """
    This function calculates the Log return and Covariance matrix for a given symbol
    """
    def ProcessSymbolData(self, symbol, F_matrix, r_t, slice: Slice):
        if symbol not in self.symbol_data:
                self.symbol_data[symbol] = {
                    "close_window_5": RollingWindow[float](5) ,
                    "close_window_22": RollingWindow[float](22) ,
                    "close_window_2": RollingWindow[float](2) ,
                    "high": RollingWindow[float](2) ,
                    "low": RollingWindow[float](2) ,
                    "wti_window": RollingWindow[float](2) ,
                    "logr": self.LOGR(symbol, 1)
                }

        data = self.symbol_data[symbol]
        slicecheck = self.current_slice
        if symbol in slicecheck and slice.Bars.ContainsKey(symbol):
            trade_bar = slice.Bars[symbol]
            close_price = trade_bar.Close
            corrected_spread = NSJ_t = five_days_vol = twentytwo_days_vol = wti_return = vix_value = None
            risk_free_interest_rate = self.risk_free_interest_rate_model.get_interest_rate(slice.time)

            # Add data to rolling windows
            data["close_window_5"].Add(close_price)
            data["close_window_22"].Add(close_price)
            data["close_window_2"].Add(close_price)
            data["high"].Add(trade_bar.High)
            data["low"].Add(trade_bar.Low)
            data['wti_window'].add(self.Securities[self.wti].Price)

            # Indexes calculation
            if data["close_window_5"].IsReady:
                five_days_vol = self.CalculateRealizedVolatility(data["close_window_5"])
                NSJ_t = self.Compute_NSJ(data["close_window_5"])

            if data['close_window_22'].IsReady:
                twentytwo_days_vol= self.CalculateRealizedVolatility(data['close_window_22'])

            if data['high'].IsReady and data['low'].IsReady and data['close_window_2'].IsReady:
                corrected_spread = self.CalculateCorrectedSpread(data['close_window_2'], data['high'], data['low'])

            if data['wti_window'].IsReady:
                wti_return = self.CalculateDailyWTIreturns(data['wti_window'])

            vix_value = self.Securities[self.vix_symbol].Price

            if (data['logr'].Current.Value != 0) and None not in [risk_free_interest_rate, corrected_spread, NSJ_t, five_days_vol, twentytwo_days_vol, vix_value, wti_return, data['logr']]:
                new_f_row = np.array([[risk_free_interest_rate, corrected_spread, NSJ_t, five_days_vol, twentytwo_days_vol, 
                                vix_value, wti_return]])
                new_logr_value = data['logr'].Current.Value
                log_r = np.array([new_logr_value])

                if F_matrix is None:
                    F_matrix = new_f_row
                else:
                    F_matrix = np.append(F_matrix, new_f_row, axis=0)

                if r_t is None:
                    r_t = log_r
                else:
                    r_t = np.append(r_t, log_r, axis=0)
        else:
            self.Log(f"current stock {symbol} Not In Slice")
        return F_matrix, r_t

    """
    This function calculates the realized Volatility given a slicing window
    """
    def CalculateRealizedVolatility(self, close_window):
        # Convert prices to log returns
        log_returns = [np.log(close_window[i+1] / close_window[i]) for i in range(close_window.Count - 1)]
        # Calculate the standard deviation of log returns
        volatility = np.std(log_returns)
        return volatility

    """
    This function calculates the corrected spread given a slicing window
    """
    def CalculateCorrectedSpread(self, close, high, low):
        sum_spread = 0
        for i in range(1):
            ct = np.log(close[i])
            ct_next = np.log(close[i+1])

            eta_t = np.log((high[i] + low[i]) / 2)
            eta_next = np.log((high[i+1] + low[i+1]) / 2)
            
            inner_expression = 4 * (ct - eta_t) * (ct_next - eta_next)
            st = np.sqrt(max(inner_expression, 0))
            sum_spread += st
        # Calculate the average of the corrected spreads
        two_day_corrected_spread = sum_spread / 2
        return two_day_corrected_spread
    
    """
    This function calculates the WTI return given a slicing window
    """
    def CalculateDailyWTIreturns(self,wti_window):
        if wti_window[0] == 0:
            e = 1e-18
            wti_return = (wti_window[1]-wti_window[0])/e
        else:
            wti_return = (wti_window[1]-wti_window[0])/wti_window[0]
        return wti_return

    """
    This function calculates the NSJ given a slicing window
    """
    def Compute_NSJ(self,window):
        daily_return = [np.log(window[i+1] / window[i]) for i in range(window.Count - 1)]
        rst_minus = sum(x ** 2 for x in daily_return if x <= 0)
        bpv = 0
        for i in range(1,4):
            bpv += abs(daily_return[i] * daily_return[i-1])
        bpv *= (np.pi / 2)
        nsj = rst_minus - (0.5*bpv)
        return nsj