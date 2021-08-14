import pytest
from gmocoin_backtest import Backtest


@pytest.fixture(scope="module", autouse=True)
def scope_module():
    class MyBacktest(Backtest):
        def strategy(self):
            fast_ma = self.ema(period=3)
            slow_ma = self.ema(period=5)
            self.sell_exit = self.buy_entry = (fast_ma > slow_ma) & (
                fast_ma.shift() <= slow_ma.shift()
            )
            self.buy_exit = self.sell_entry = (fast_ma < slow_ma) & (
                fast_ma.shift() >= slow_ma.shift()
            )
            self.size = 0.1
            self.stop_loss = 5
            self.take_profit = 10

    yield MyBacktest(
        symbol="BTC_JPY",
        sqlite_file_name="backtest.sqlite3",
        from_date="2021-07-15",
        to_date="2021-08-15",
        interval="1T",
        data_dir="data",
    )


@pytest.fixture(scope="function", autouse=True)
def backtest(scope_module):
    yield scope_module


# @pytest.mark.skip
def test_backtest(backtest):
    backtest.run()
