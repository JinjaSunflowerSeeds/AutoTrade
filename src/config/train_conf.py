class FeatureList:
    def __init__(self):
        self.dropout_features = (
            [
                # 'stock splits',
                # 'star',
                # 'sin_hour',
                # 'shooting_star',
                # 'shares-outstanding',
                # 'rsi_singal_v2',
                # 'rsi_boll_vwap_signal',
                # 'rain_drop_doji',
                # 'rain_drop',
                # 'quarter',
                # 'pre-tax-income',
                # 'piercing_pattern',
                # 'net-income',
                # 'morning_star_doji',
                # 'morning_star',
                # 'month',
                # 'mfi_signal',
                # 'macd_rsi',
                # 'ma_signal',
                # 'kdjk_slope_dir',
                # 'kdjk_slope',
                # 'ifco',
                # 'hanging_man',
                # 'gravestone_doji',
                # 'evening_star_doji',
                # 'eps-diluted',
                # 'ebitda',
                # 'ebit',
                # 'dragonfly_doji',
                # 'doji_star',
                # 'dividends',
                # 'dark_cloud_cover',
                # 'cos_quarter',
                # 'cos_hour','stock splits', 'sin_hour', 'shooting_star', 'shares-outstanding', 'rsi_singal_v2', 'rsi_boll_vwap_signal', 'rain_drop_doji', 'quarter', 'pre-tax-income', 'piercing_pattern', 'net-income', 'morning_star_doji', 'month', 'macd_rsi', 'ma_signal', 'kdjk_slope_dir', 'kdjk_slope', 'ifco', 'hanging_man', 'gravestone_doji', 'gdp', 'evening_star_doji', 'eps-diluted', 'eps', 'ebitda', 'ebit', 'dragonfly_doji', 'doji_star', 'dividends', 'cos_quarter',
            ],
        )
        self._fd = {
            "ohlcv": [
                "open",
                "high",
                "low",
                "close",
                "volume",
                #   , "next_open"
            ],
            # "correlated_stocks": [  # other releveant stocks
            #     "^hsi"	,"hyg"	,	"amd"	,"frgt",	"dji",	"aapl",	"^vix",	"tsm",	"xela",	"xlk",	"jpy=x",	"hpq",	"avgo",	"spy",	"intc",	"cohr","msft"
            # ],
            # "trading_signals": [  # trading singals
            #     "macd_crossover",
            #     "rsi_singal",
            #     "rsi_singal_v2",
            #     "macd_rsi",
            #     "mfi_signal",
            #     "ma_signal",
            #     "rsi_boll_signal",
            #     "vwap_signal",
            #     "rsi_boll_vwap_signal",
            #     "ichimoku_signal",
            #     "kdjk_singal",
            #     "kdjk_singal_v2",
            #     "kdjk_singal_v3",
            #     #
            #     "close_slope",
            #     "close_slope_dir",
            #     "kdjk_slope",
            #     "kdjk_slope_dir",
            # ],
            # "economic_data": [
            #     "cpi",
            #     "mortgage",
            #     "score",
            #     "unemployment",
            #     "gdp",
            #     # 'treasury',
            #     # 'rating',
            # ],
            # "fundementals": [
            #     "stock splits",
            #     "dividends",
            #     "shares",
            #     "eps estimate",
            #     "reported eps",
            #     "surprise(%)",
            # ],
            # "engineered": [
            #     # derived by me
            #     "sin_quarter",
            #     "sin_month",
            #     "sin_week",
            #     "sin_day_of_year",
            #     "sin_hour",
            #     "cos_quarter",
            #     "cos_month",
            #     "cos_week",
            #     "cos_day_of_year",
            #     "cos_hour",
            #     "quarter",
            #     "month",
            #     "week",
            #     "day_of_year",
            #     "up_cnt_7d",
            #     "up_cnt_30d",
            #     "high_price_7d",
            #     "high_price_30d",
            #     "low_price_7d",
            #     "low_price_30d",
            #     "tslHp",
            #     "tslLp",
            #     "slope_3d",
            #     "slope_7d",
            #     "slope_30d",
            #     "poly_3d",
            #     "poly_7d",
            #     "poly_30d"
            #      "vwap"
            # ],
            # "fundementals from macrotrends": [
            #     "revenue",
            #     "cost",
            #     "g-profit",
            #     "rnd",
            #     "sgae",
            #     "operating-expenses",
            #     "operating-income",
            #     "tnoie",
            #     "pre-tax-income",
            #     "tpit",
            #     "income-after-taxes",
            #     "ifco",
            #     "net-income",
            #     "ebitda",
            #     "ebit",
            #     "basic-shares-outstanding",
            #     "shares-outstanding",
            #     "eps",
            #     "eps-diluted",
            # ],
            # "technicals": [
            #     "middle",
            #     "log-ret",
            #     # "up",
            #     "up_10_c",
            #     "rsi_6",
            #     "rsi_14",
            #     "rsi_26",
            #     "stochrsi_6",
            #     "stochrsi_14",
            #     "stochrsi_26",
            #     "wt1",
            #     "wt2",
            #     "close_7_smma",
            #     "close_14_smma",
            #     "close_21_smma",
            #     "close_10_roc",
            #     "close_21_roc",
            #     "close_5_mad",
            #     "close_10_mad",
            #     "close_25_mad",
            #     "close_12_trix",
            #     "close_12_tema",
            #     # "change",
            #     "vr_26",
            #     "wr_14",
            #     "cci_14",
            #     "atr_14",
            #     "supertrend_ub",
            #     "supertrend_lb",
            #     "supertrend",
            #     "dma",
            #     "pdi",
            #     "ndi",
            #     "dx",
            #     "adx",
            #     "adxr",
            #     "kdjk",
            #     "kdjd",
            #     "kdjj",
            #     "cr",
            #     "cr-ma1",
            #     "cr-ma2",
            #     "cr-ma3",
            #     "boll_20",
            #     "boll_ub_20",
            #     "boll_lb_20",
            #     "macd",
            #     "macds",
            #     "macdh",
            #     "ppo",
            #     "ppos",
            #     "ppoh",
            #     "vwma_14",
            #     "close_14_sma",
            #     "close_14_mstd",
            #     "chop_14",
            #     "mfi_14",
            #     "eribull",
            #     "eribear",
            #     "eribull_5",
            #     "eribear_5",
            #     "close_10_ker",
            #     "close_10,2,30_kama",
            #     "aroon_14",
            #     "ao",
            #     "bop",
            #     "cmo",
            #     "coppock",
            #     "ichimoku",
            #     "close_10_lrma",
            #     "cti",
            #     "ftr",
            #     "rvgi",
            #     "rvgis",
            #     "rvgi_5",
            #     "rvgis_5",
            #     "inertia",
            #     "kst",
            #     "pgo",
            #     "psl",
            #     "pvo",
            #     "pvos",
            #     "pvoh",
            #     "qqe",
            #     "qqel",
            #     "qqes",
            # ],
            "candlestick_v2": [
                "cs_2crows",
                "cs_3blackcrows",
                "cs_3inside",
                "cs_3linestrike",
                "cs_3outside",
                "cs_3starsinsouth",
                "cs_3whitesoldiers",
                "cs_abandonedbaby",
                "cs_advanceblock",
                "cs_belthold",
                "cs_breakaway",
                "cs_closingmarubozu",
                "cs_concealbabyswall",
                "cs_counterattack",
                "cs_darkcloudcover",
                "cs_doji_10_0.1",
                "cs_dojistar",
                "cs_dragonflydoji",
                "cs_engulfing",
                "cs_eveningdojistar",
                "cs_eveningstar",
                "cs_gapsidesidewhite",
                "cs_gravestonedoji",
                "cs_hammer",
                "cs_hangingman",
                "cs_harami",
                "cs_haramicross",
                "cs_highwave",
                "cs_hikkake",
                "cs_hikkakemod",
                "cs_homingpigeon",
                "cs_identical3crows",
                "cs_inneck",
                "cs_inside",
                "cs_invertedhammer",
                "cs_kicking",
                "cs_kickingbylength",
                "cs_ladderbottom",
                "cs_longleggeddoji",
                "cs_longline",
                "cs_marubozu",
                "cs_matchinglow",
                "cs_mathold",
                "cs_morningdojistar",
                "cs_morningstar",
                "cs_onneck",
                "cs_piercing",
                "cs_rickshawman",
                "cs_risefall3methods",
                "cs_separatinglines",
                "cs_shootingstar",
                "cs_shortline",
                "cs_spinningtop",
                "cs_stalledpattern",
                "cs_sticksandwich",
                "cs_takuri",
                "cs_tasukigap",
                "cs_thrusting",
                "cs_tristar",
                "cs_unique3river",
                "cs_upsidegap2crows",
                "cs_xsidegap3methods",
            ],
            # "candlestick_v0":[
            #     "BullishHorn",
            #     "BearHorn",
            #     "BullishHigh",
            #     "BearHigh",
            #     "BullishLow",
            #     "BearLow",
            #     "BullishHarami",
            #     "BearHarami",
            #     "inverted_hammer",
            #     "bearish_engulfing",
            #     "dark_cloud_cover",
            #     "evening_star_doji",
            #     "morning_star",
            #     "shooting_star",
            #     "bearish_harami",
            #     "doji",
            #     "gravestone_doji",
            #     "morning_star_doji",
            #     "star",
            #     "bullish_engulfing",
            #     "doji_star",
            #     "hammer",
            #     "piercing_pattern",
            #     "bullish_harami",
            #     "dragonfly_doji",
            #     "hanging_man",
            #     "rain_drop",
            #     "evening_star",
            #     "rain_drop_doji",
            # ],
        }
        self.features = [
            j for i in self._fd.values() for j in i if j not in self.dropout_features
        ]
        # self.features = [ 'bearish_harami', 'cr-ma1', 'day_of_year', 'unemployment', 'aroon_14', 'coppock', 'macdh', 'rsi_26', 'tpit', 'ppos', 'volume', 'ao', 'low_price_30d', 'dji', 'kdjk_singal_v2', 'hpq', 'slope_3d', 'mortgage', 'supertrend', 'doji', 'sin_month', 'bop', 'stochrsi_6'
        #  ]


class TrainConf:
    def __init__(self):
        self.do_binary = True
        self.threshold = 0.1
        self.direction = "buy"  # buy, sell
        self.start_from = 30
        self.pca_n_components = 0.99
        self.do_pca = False
