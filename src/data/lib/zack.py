from __future__ import print_function
import time
import intrinio_sdk as intrinio
from intrinio_sdk.rest import ApiException
import pandas as pd

intrinio.ApiClient().set_api_key('Ojk1YmQwY2E1NmZmZTRhNDQzOTQ0MTdmNjc3NzgxZGJm')
intrinio.ApiClient().allow_retries(True)

identifier = 'AAPL'
page_size = 10000
next_page = ''

response = intrinio.ZacksApi().get_zacks_analyst_ratings(
    identifier=identifier, page_size=page_size, next_page=next_page,
    start_date='2013-01-01', end_date='2023-12-31')
print(response)


print(pd.DataFrame(response))
