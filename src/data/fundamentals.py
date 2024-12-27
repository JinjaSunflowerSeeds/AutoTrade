import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta

sys.path.append("./")
import os
from datetime import datetime

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from joblib import delayed, Parallel
from utils.config_reader import get_fundamentals_data_conf
from utils.logging import LoggerUtility

yf.pdr_override()
logger = LoggerUtility.setup_logger(__file__)


class HistoricalPE:
    def __init__(self, ticker, company, filepath):
        """
        Scrapes financial statements from Macrotrends.

        Args:
            ticker (str): Stock ticker symbol.
            company (str): Company name.
            filepath (str): Filepath template for output files.
        """
        # self.url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company}/financial-statements?freq=Q"
        self.url = f"https://www.macrotrends.net/stocks/charts/{ticker}/{company.lower()}/income-statement?freq=Q"
        self.ticker = ticker
        self.filepath = filepath.format(ticker)

    def scrape(self):
        """Scrape data from Macrotrends."""
        try:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36"
                }
            )
            response = session.get(self.url)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Error during scraping: {e}")
            return None

    def parse_data(self, response):
        """Extract and transform scraped data into a DataFrame."""
        try:
            # Extract numerical and textual data using regex
            num = re.findall(r"(?<=div\>\"\,)[0-9\.\"\:\-\, ]*", response.text)
            text = re.findall(r"(?<=s\: \')\S+(?=\'\, freq)", response.text)

            # Convert to dictionary
            data_dicts = [json.loads(f"{{{entry}}}") for entry in num]

            # Create DataFrame
            df = pd.DataFrame(
                {name: dict_.values() for name, dict_ in zip(text, data_dicts)}
            )
            df.index = next(iter(data_dicts)).keys()
            # Rename columns for clarity
            column_mapping = {
                "cost-goods-sold": "cost_of_goods_sold",
                "gross-profit": "gross_profit",
                "research-development-expenses": "r_and_d_expenses",
                "selling-general-administrative-expenses": "sga_expenses",
                "total-non-operating-income-expense": "non_operating_income",
                "total-provision-income-taxes": "income_taxes",
                "income-from-continuous-operations": "operating_income",
                "eps-basic-net-earnings-per-share": "eps_basic",
                "eps-earnings-per-share-diluted": "eps_diluted",
            }
            df.rename(columns=column_mapping, inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error during data parsing: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df):
        try:
            logger.info(f"Saving data to {self.filepath}")
            filepath = self.filepath + "/financial"
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath += "/{}_to_{}.csv".format(
                df.index.min(),
                df.index.max(),
            )
            df.index.name = "date"
            df.to_csv(filepath)
            logger.info(f"Data successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def run(self):
        """Main method to scrape, parse, and save data."""
        logger.info(f"Scraping data from {self.url}...")
        response = self.scrape()
        if response:
            df = self.parse_data(response)
            if not df.empty:
                self.save_to_csv(df)
            else:
                logger.error("Parsed data is empty.")
        else:
            logger.error("Failed to retrieve data.")


class HistoricalData:
    def __init__(self, ticker, start_date, end_date, filepath):
        """
        Fetch historical data for a stock ticker.

        Args:
            ticker (str): Stock ticker symbol.
            start_date (datetime): Start date for historical data.
            end_date (datetime): End date for historical data.
            filepath (str): Filepath template for output files.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.filepath = filepath.format(ticker)
        self.data = {}

    def fetch_data(self):
        """Fetch historical data using yfinance."""
        try:

            tk = yf.Ticker(self.ticker)
            self.data["dividends"] = (
                tk.dividends
            )  # pd.DataFrame(tk.actions[tk.actions['Stock Splits']==0]['Dividends'], columns=['Dividends'])
            self.data["splits"] = (
                tk.splits
            )  # pd.DataFrame(tk.actions[tk.actions['Stock Splits']>0]['Stock Splits'], columns=['Stock Splits'])
            self.data["share_count"] = pd.DataFrame(
                tk.get_shares_full(start=self.start_date, end=self.end_date),
                columns=["shares"],
            )
            self.data["quarterly_income_stmt"] = tk.quarterly_income_stmt.T
            self.data["balance_sheet"] = tk.balance_sheet.T
            self.data["cashflow"] = tk.cashflow.T
            self.data["earnings_dates"] = tk.get_earnings_dates(limit=40000)
            self.data["major_holders"] = pd.DataFrame(
                tk.major_holders.values, columns=["Share%", "Name"]
            )
            self.data["institutional_holders"] = tk.institutional_holders
            self.data["mutualfund_holders"] = tk.mutualfund_holders
            self.data['major_holders']['date'] = datetime.now().strftime("%Y-%m-%d")
            self.data['institutional_holders']['date'] = datetime.now().strftime("%Y-%m-%d")
            self.data['mutualfund_holders']['date'] = datetime.now().strftime("%Y-%m-%d")
            self.data['quarterly_income_stmt'].index.name = 'date'
            self.data['share_count'].index.name = 'date'
            self.data['dividends'].index.name = 'date'
            self.data['splits'].index.name = 'date'
            self.data['balance_sheet'].index.name = 'date'
            self.data['cashflow'].index.name = 'date'
            opt_expration = pd.DataFrame(
                tk.options, columns=["expiration"]
            ).sort_values("expiration")
            calls = pd.DataFrame()
            puts = pd.DataFrame()
            for e in opt_expration.expiration.values:
                try:
                    opt = tk.option_chain(e)
                    call, put = opt.calls, opt.puts
                    call["expirationDate"],call["date"], put["expirationDate"],put['date'] = e, e,e,e
                    call["type"], put["type"] = "call", "put"
                    calls = pd.concat([calls, call], ignore_index=True)
                    puts = pd.concat([puts, put], ignore_index=True)
                except Exception as er:
                    logger.info(er)
                    continue
            self.data["options"] = pd.concat([calls, puts], ignore_index=True)
        except Exception as e:
            logger.info(f"Error fetching tk  data: {e}")

    def get_file_path(self, key, df):
        if key in [
            "major_holders",
            "institutional_holders",
            "mutualfund_holders",
        ]:
            return "/{}.csv".format(datetime.now().strftime("%Y_%m_%d"))
        elif key in ["options"]:
            return "/{}_to_{}.csv".format(
                df.expirationDate.min(), df.expirationDate.max()
            )
        else:
            return "/{}_to_{}.csv".format(
                df.index.min().strftime("%Y_%m_%d"),
                df.index.max().strftime("%Y_%m_%d"),
            )

    def save_to_csv(self):
        for key, df in self.data.items():
            if df is not None and df.empty is False:
                filepath = self.filepath + f"/{key}"
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                filepath += self.get_file_path(key, df)
                df.to_csv(filepath)
                logger.info(f"Saved {key} to {filepath}")

    def run(self):
        """Main method to fetch and save data."""
        logger.info(f"Fetching historical data for {self.ticker}...")
        self.fetch_data()
        self.save_to_csv()


if __name__ == "__main__":
    ticker, company, data_output_base_dir, lookback = get_fundamentals_data_conf()

    # Historical PE
    pe_scraper = HistoricalPE(
        ticker=ticker, company=company, filepath=data_output_base_dir
    )
    pe_scraper.run()

    # Historical Data
    start_date = datetime.now() - relativedelta(years=10)
    end_date = datetime.now()
    historical_data = HistoricalData(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        filepath=data_output_base_dir,
    )
    historical_data.run()
